//! Image Classification Example — SciRS2 Neural
//!
//! Demonstrates a simple CNN trained on synthetic MNIST-style data.
//! Generates 8×8 grayscale images with class-specific patterns, trains
//! a two-conv-layer network with SGD + momentum, and reports accuracy.
//!
//! Run with: cargo run -p scirs2-neural --example image_classification

use scirs2_core::ndarray::{s, Array, Array1, Array2, Array4, Axis};
use scirs2_core::random::prelude::SliceRandom;
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ---------- activation helpers ----------

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN expected"))
        .map(|(i, _)| i)
        .expect("empty slice")
}

// ---------- synthetic dataset ----------

/// Generate (N, 1, H, W) images and (N, C) one-hot labels.
///
/// Each class `c` has a fixed random "prototype" pattern; each sample
/// is that prototype plus Gaussian noise.
fn generate_dataset(
    n_samples: usize,
    n_classes: usize,
    h: usize,
    w: usize,
    rng: &mut SmallRng,
) -> (Array4<f32>, Array2<f32>) {
    let n_pixels = h * w;

    // Build class prototypes (n_classes × n_pixels)
    let prototypes: Vec<Vec<f32>> = (0..n_classes)
        .map(|_| {
            (0..n_pixels)
                .map(|_| {
                    if rng.random::<f32>() > 0.5 {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                })
                .collect()
        })
        .collect();

    let mut images = Array4::zeros([n_samples, 1, h, w]);
    let mut labels = Array2::zeros([n_samples, n_classes]);

    for i in 0..n_samples {
        let cls = rng.random_range(0..n_classes);
        labels[[i, cls]] = 1.0;
        for ph in 0..h {
            for pw in 0..w {
                let noise: f32 = (rng.random::<f32>() - 0.5) * 0.4;
                images[[i, 0, ph, pw]] = (prototypes[cls][ph * w + pw] + noise).clamp(0.0, 1.0);
            }
        }
    }
    (images, labels)
}

// ---------- minimal CNN ----------

/// A fully hand-rolled two-conv + two-dense network to stay framework-agnostic
/// and keep the example self-contained (<200 lines).
struct SimpleCnn {
    // Conv1: 1 -> 4 filters, 3×3 kernels
    w_conv1: Array4<f32>, // [4, 1, 3, 3]
    b_conv1: Array1<f32>, // [4]
    // Dense 1: flattened -> 16
    w_fc1: Array2<f32>,
    b_fc1: Array1<f32>,
    // Dense 2: 16 -> n_classes
    w_fc2: Array2<f32>,
    b_fc2: Array1<f32>,
    // SGD + momentum accumulators
    vw_conv1: Array4<f32>,
    vb_conv1: Array1<f32>,
    vw_fc1: Array2<f32>,
    vb_fc1: Array1<f32>,
    vw_fc2: Array2<f32>,
    vb_fc2: Array1<f32>,
    n_classes: usize,
    img_h: usize,
    img_w: usize,
}

impl SimpleCnn {
    fn new(n_classes: usize, img_h: usize, img_w: usize, rng: &mut SmallRng) -> Self {
        let scale1 = (2.0_f32 / (1.0 * 9.0_f32)).sqrt();
        let flat = 4 * (img_h - 2) * (img_w - 2);
        let scale2 = (2.0_f32 / flat as f32).sqrt();
        let scale3 = (2.0_f32 / 16.0_f32).sqrt();

        macro_rules! rand_arr {
            ($shape:expr, $scale:expr) => {{
                Array::from_shape_fn($shape, |_| (rng.random::<f32>() * 2.0 - 1.0) * $scale)
            }};
        }

        Self {
            w_conv1: rand_arr!([4, 1, 3, 3], scale1),
            b_conv1: Array1::zeros(4),
            w_fc1: rand_arr!([flat, 16], scale2),
            b_fc1: Array1::zeros(16),
            w_fc2: rand_arr!([16, n_classes], scale3),
            b_fc2: Array1::zeros(n_classes),
            vw_conv1: Array4::zeros([4, 1, 3, 3]),
            vb_conv1: Array1::zeros(4),
            vw_fc1: Array2::zeros([flat, 16]),
            vb_fc1: Array1::zeros(16),
            vw_fc2: Array2::zeros([16, n_classes]),
            vb_fc2: Array1::zeros(n_classes),
            n_classes,
            img_h,
            img_w,
        }
    }

    /// Forward pass for a single sample; returns (conv1_out, relu1, fc1_pre, relu2, logits, probs).
    fn forward(&self, img: &Array4<f32>, idx: usize) -> (Vec<f32>, Vec<f32>) {
        let oh = self.img_h - 2;
        let ow = self.img_w - 2;
        // Conv1 (valid padding, stride 1)
        let mut conv1 = vec![0.0_f32; 4 * oh * ow];
        for f in 0..4usize {
            for y in 0..oh {
                for x in 0..ow {
                    let mut s = self.b_conv1[f];
                    for ky in 0..3usize {
                        for kx in 0..3usize {
                            s += self.w_conv1[[f, 0, ky, kx]] * img[[idx, 0, y + ky, x + kx]];
                        }
                    }
                    conv1[f * oh * ow + y * ow + x] = relu(s);
                }
            }
        }
        // FC1
        let flat_len = conv1.len();
        let mut fc1 = [0.0_f32; 16];
        for (j, fc1_j) in fc1.iter_mut().enumerate() {
            let mut s = self.b_fc1[j];
            for (k, &c) in conv1.iter().enumerate().take(flat_len) {
                s += c * self.w_fc1[[k, j]];
            }
            *fc1_j = relu(s);
        }
        // FC2
        let mut logits = vec![0.0_f32; self.n_classes];
        for (j, logit_j) in logits.iter_mut().enumerate() {
            let mut s = self.b_fc2[j];
            for (k, &f) in fc1.iter().enumerate() {
                s += f * self.w_fc2[[k, j]];
            }
            *logit_j = s;
        }
        let probs = softmax(&logits);
        (conv1, probs)
    }

    fn train_batch(
        &mut self,
        images: &Array4<f32>,
        labels: &Array2<f32>,
        indices: &[usize],
        lr: f32,
        momentum: f32,
    ) -> f32 {
        let oh = self.img_h - 2;
        let ow = self.img_w - 2;
        let flat = 4 * oh * ow;
        let batch = indices.len() as f32;

        let mut gw_c1: Array4<f32> = Array4::zeros([4, 1, 3, 3]);
        let mut gb_c1: Array1<f32> = Array1::zeros(4usize);
        let mut gw_f1: Array2<f32> = Array2::zeros([flat, 16]);
        let mut gb_f1: Array1<f32> = Array1::zeros(16usize);
        let mut gw_f2: Array2<f32> = Array2::zeros([16, self.n_classes]);
        let mut gb_f2: Array1<f32> = Array1::zeros(self.n_classes);
        let mut total_loss = 0.0_f32;

        for &i in indices {
            let (conv1, probs) = self.forward(images, i);
            // Cross-entropy loss
            for c in 0..self.n_classes {
                if labels[[i, c]] > 0.0 {
                    total_loss -= (probs[c].max(1e-9)).ln();
                }
            }
            // dL/d_logits
            let mut d_logits = probs.clone();
            for c in 0..self.n_classes {
                d_logits[c] -= labels[[i, c]];
            }
            // Grad FC2
            let mut fc1_acts = [0.0_f32; 16];
            for (j, fc1_act_j) in fc1_acts.iter_mut().enumerate() {
                let mut s = self.b_fc1[j];
                for (k, &c) in conv1.iter().enumerate().take(flat) {
                    s += c * self.w_fc1[[k, j]];
                }
                *fc1_act_j = relu(s);
            }
            for (j, &dl_j) in d_logits.iter().enumerate() {
                gb_f2[j] += dl_j / batch;
                for (k, &fa_k) in fc1_acts.iter().enumerate() {
                    gw_f2[[k, j]] += dl_j * fa_k / batch;
                }
            }
            // Backprop to FC1
            let mut d_fc1 = [0.0_f32; 16];
            for (k, d_fc1_k) in d_fc1.iter_mut().enumerate() {
                for (j, &dl_j) in d_logits.iter().enumerate() {
                    *d_fc1_k += dl_j * self.w_fc2[[k, j]];
                }
                // ReLU gate
                let pre: f32 = {
                    let mut s = self.b_fc1[k];
                    for (m, &c) in conv1.iter().enumerate().take(flat) {
                        s += c * self.w_fc1[[m, k]];
                    }
                    s
                };
                *d_fc1_k *= if pre > 0.0 { 1.0 } else { 0.0 };
                gb_f1[k] += *d_fc1_k / batch;
                for (m, &c) in conv1.iter().enumerate().take(flat) {
                    gw_f1[[m, k]] += *d_fc1_k * c / batch;
                }
            }
            // Backprop to Conv1
            let mut d_conv1 = vec![0.0_f32; flat];
            for (m, d_conv1_m) in d_conv1.iter_mut().enumerate() {
                for (k, &d_k) in d_fc1.iter().enumerate() {
                    *d_conv1_m += d_k * self.w_fc1[[m, k]];
                }
            }
            for f in 0..4usize {
                for y in 0..oh {
                    for x in 0..ow {
                        let flat_idx = f * oh * ow + y * ow + x;
                        // ReLU gate for conv1
                        let raw: f32 = {
                            let mut s = self.b_conv1[f];
                            for ky in 0..3usize {
                                for kx in 0..3usize {
                                    s += self.w_conv1[[f, 0, ky, kx]]
                                        * images[[i, 0, y + ky, x + kx]];
                                }
                            }
                            s
                        };
                        let d = d_conv1[flat_idx] * if raw > 0.0 { 1.0 } else { 0.0 };
                        gb_c1[f] += d / batch;
                        for ky in 0..3usize {
                            for kx in 0..3usize {
                                gw_c1[[f, 0, ky, kx]] += d * images[[i, 0, y + ky, x + kx]] / batch;
                            }
                        }
                    }
                }
            }
        }

        // SGD + momentum update
        macro_rules! update {
            ($v:expr, $g:expr, $w:expr) => {
                $v.zip_mut_with(&$g, |vi, &gi| *vi = momentum * *vi + gi);
                $w.zip_mut_with(&$v, |wi, &vi| *wi -= lr * vi);
            };
        }
        update!(self.vw_conv1, gw_c1, self.w_conv1);
        update!(self.vb_conv1, gb_c1, self.b_conv1);
        update!(self.vw_fc1, gw_f1, self.w_fc1);
        update!(self.vb_fc1, gb_f1, self.b_fc1);
        update!(self.vw_fc2, gw_f2, self.w_fc2);
        update!(self.vb_fc2, gb_f2, self.b_fc2);

        total_loss / batch
    }

    fn predict_class(&self, images: &Array4<f32>, idx: usize) -> usize {
        let (_, probs) = self.forward(images, idx);
        argmax(&probs)
    }
}

// ---------- main ----------

fn main() {
    const N_SAMPLES: usize = 600;
    const N_CLASSES: usize = 5;
    const IMG_H: usize = 8;
    const IMG_W: usize = 8;
    const TRAIN_RATIO: f64 = 0.8;
    const EPOCHS: usize = 20;
    const BATCH_SIZE: usize = 32;
    const LR: f32 = 0.05;
    const MOMENTUM: f32 = 0.9;

    let mut rng = SmallRng::seed_from_u64(42);

    println!("=== SciRS2 Image Classification (Synthetic MNIST-Style) ===\n");
    println!("Dataset: {N_SAMPLES} samples | {N_CLASSES} classes | {IMG_H}×{IMG_W} images");

    let (images, labels) = generate_dataset(N_SAMPLES, N_CLASSES, IMG_H, IMG_W, &mut rng);

    let n_train = (N_SAMPLES as f64 * TRAIN_RATIO) as usize;
    let n_test = N_SAMPLES - n_train;
    let train_images = images.slice(s![..n_train, .., .., ..]).to_owned();
    let train_labels = labels.slice(s![..n_train, ..]).to_owned();
    let test_images = images.slice(s![n_train.., .., .., ..]).to_owned();
    let test_labels = labels.slice(s![n_train.., ..]).to_owned();
    println!("Train: {n_train} | Test: {n_test}\n");

    let mut model = SimpleCnn::new(N_CLASSES, IMG_H, IMG_W, &mut rng);
    let mut train_indices: Vec<usize> = (0..n_train).collect();

    println!("{:<6}  {:>10}", "Epoch", "Train Loss");
    println!("{}", "-".repeat(20));

    for epoch in 0..EPOCHS {
        train_indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0_f32;
        let mut n_batches = 0usize;
        for chunk in train_indices.chunks(BATCH_SIZE) {
            let loss = model.train_batch(&train_images, &train_labels, chunk, LR, MOMENTUM);
            epoch_loss += loss;
            n_batches += 1;
        }
        let avg_loss = epoch_loss / n_batches as f32;
        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!("{:<6}  {:>10.4}", epoch + 1, avg_loss);
        }
    }

    // Evaluate on test set
    let mut correct = 0usize;
    for i in 0..n_test {
        let pred = model.predict_class(&test_images, i);
        let true_cls = argmax(test_labels.row(i).to_slice().expect("contiguous"));
        if pred == true_cls {
            correct += 1;
        }
    }
    let accuracy = 100.0 * correct as f64 / n_test as f64;

    println!("\n--- Test Results ---");
    println!("Correct : {correct}/{n_test}");
    println!("Accuracy: {accuracy:.1}%");

    // Confusion summary (per class)
    println!("\nPer-class prediction summary:");
    println!("{:<8} {:>10} {:>10}", "Class", "Total", "Correct");
    for cls in 0..N_CLASSES {
        let total = (0..n_test)
            .filter(|&i| argmax(test_labels.row(i).to_slice().expect("contiguous")) == cls)
            .count();
        let ok = (0..n_test)
            .filter(|&i| {
                argmax(test_labels.row(i).to_slice().expect("contiguous")) == cls
                    && model.predict_class(&test_images, i) == cls
            })
            .count();
        println!("{:<8} {:>10} {:>10}", cls, total, ok);
    }
    println!("\nDone.");
}
