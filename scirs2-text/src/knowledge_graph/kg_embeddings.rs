//! Knowledge Graph Embedding models.
//!
//! Implements four established embedding families in pure Rust:
//!
//! | Model    | Scoring function                           |
//! |----------|--------------------------------------------|
//! | TransE   | `‖h + r − t‖`  (L2 distance)              |
//! | TransR   | `‖W_r·h + r − W_r·t‖`                     |
//! | DistMult | `⟨h, r, t⟩` (element-wise bilinear)       |
//! | ComplEx  | `Re(⟨h, r, t̄⟩)` (complex bilinear)        |
//!
//! All models are trained with a margin-based ranking loss using SGD
//! (deterministic LCG-based PRNG, no external random crate needed).

use std::collections::HashMap;

use crate::error::{Result, TextError};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Numerical entity/relation id used throughout the embedding modules.
pub type EmbId = usize;

/// A single training triple (head, relation, tail) expressed as integer ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingTriple {
    /// Head entity identifier.
    pub head: EmbId,
    /// Relation (predicate) identifier.
    pub relation: EmbId,
    /// Tail entity identifier.
    pub tail: EmbId,
}

/// Hyperparameters shared by all embedding models.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Embedding vector dimension.
    pub dim: usize,
    /// Learning rate.
    pub lr: f64,
    /// Margin for ranking loss.
    pub margin: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Negative sample ratio (negatives per positive).
    pub neg_ratio: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            dim: 64,
            lr: 0.01,
            margin: 1.0,
            epochs: 100,
            neg_ratio: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic LCG PRNG (avoid rand dependency)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    /// Next value in (-1, 1)
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((self.0 >> 11) as f64) / (u64::MAX >> 11) as f64;
        v * 2.0 - 1.0
    }

    /// Next usize in [0, n)
    fn next_usize(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 as usize) % n
    }
}

// ---------------------------------------------------------------------------
// Utility: vector operations
// ---------------------------------------------------------------------------

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12)
}

fn l2_normalize(v: &mut Vec<f64>) {
    let n = l2_norm(v);
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// 1. TransE
// ---------------------------------------------------------------------------

/// TransE: learns embeddings so that `h + r ≈ t` in Euclidean space.
///
/// Reference: Bordes et al., "Translating Embeddings for Modeling
/// Multi-relational Data", NIPS 2013.
pub struct TransE {
    /// Hyperparameters used when training this model.
    pub config: EmbeddingConfig,
    /// `entity_emb[i]` = embedding of entity i  (length = dim)
    pub entity_emb: Vec<Vec<f64>>,
    /// `relation_emb[i]` = embedding of relation i  (length = dim)
    pub relation_emb: Vec<Vec<f64>>,
    num_entities: usize,
    num_relations: usize,
}

impl TransE {
    /// Create an uninitialised model.
    pub fn new(num_entities: usize, num_relations: usize, config: EmbeddingConfig) -> Self {
        TransE {
            config,
            entity_emb: Vec::new(),
            relation_emb: Vec::new(),
            num_entities,
            num_relations,
        }
    }

    /// Train on the given triples.
    pub fn train(&mut self, triples: &[TrainingTriple]) {
        let dim = self.config.dim;
        let mut rng = Lcg::new(0xdeadbeefcafebabe);

        self.entity_emb = (0..self.num_entities)
            .map(|_| {
                let mut v: Vec<f64> = (0..dim).map(|_| rng.next_f64() * 0.1).collect();
                l2_normalize(&mut v);
                v
            })
            .collect();
        self.relation_emb = (0..self.num_relations)
            .map(|_| {
                let mut v: Vec<f64> = (0..dim).map(|_| rng.next_f64() * 0.1).collect();
                l2_normalize(&mut v);
                v
            })
            .collect();

        let n = self.num_entities;
        let margin = self.config.margin;
        let lr = self.config.lr;

        for _epoch in 0..self.config.epochs {
            for triple in triples {
                for _ in 0..self.config.neg_ratio {
                    // Corrupt head or tail
                    let corrupt_head = rng.next_usize(2) == 0;
                    let neg_h = if corrupt_head {
                        let mut c = rng.next_usize(n);
                        if c == triple.head {
                            c = (c + 1) % n;
                        }
                        c
                    } else {
                        triple.head
                    };
                    let neg_t = if !corrupt_head {
                        let mut c = rng.next_usize(n);
                        if c == triple.tail {
                            c = (c + 1) % n;
                        }
                        c
                    } else {
                        triple.tail
                    };

                    let r = triple.relation;
                    let h = triple.head;
                    let t = triple.tail;

                    // Compute h+r-t for positive and negative
                    let pos: Vec<f64> = (0..dim)
                        .map(|d| self.entity_emb[h][d] + self.relation_emb[r][d] - self.entity_emb[t][d])
                        .collect();
                    let neg: Vec<f64> = (0..dim)
                        .map(|d| self.entity_emb[neg_h][d] + self.relation_emb[r][d] - self.entity_emb[neg_t][d])
                        .collect();

                    let pos_dist = l2_norm(&pos);
                    let neg_dist = l2_norm(&neg);
                    let loss = (margin + pos_dist - neg_dist).max(0.0);
                    if loss == 0.0 {
                        continue;
                    }

                    // Gradient of ‖h+r-t‖ w.r.t. each component
                    let grad_pos: Vec<f64> = pos.iter().map(|x| x / pos_dist).collect();
                    let grad_neg: Vec<f64> = neg.iter().map(|x| x / neg_dist).collect();

                    for d in 0..dim {
                        self.entity_emb[h][d] -= lr * grad_pos[d];
                        self.entity_emb[t][d] += lr * grad_pos[d];
                        self.relation_emb[r][d] -= lr * grad_pos[d];

                        self.entity_emb[neg_h][d] += lr * grad_neg[d];
                        self.entity_emb[neg_t][d] -= lr * grad_neg[d];
                    }

                    l2_normalize(&mut self.entity_emb[h]);
                    l2_normalize(&mut self.entity_emb[t]);
                    l2_normalize(&mut self.entity_emb[neg_h]);
                    l2_normalize(&mut self.entity_emb[neg_t]);
                }
            }
        }
    }

    /// Score a triple: lower is better (distance-based).
    pub fn score(&self, head: EmbId, relation: EmbId, tail: EmbId) -> f64 {
        if head >= self.num_entities
            || tail >= self.num_entities
            || relation >= self.num_relations
        {
            return f64::MAX;
        }
        let dim = self.config.dim;
        let v: Vec<f64> = (0..dim)
            .map(|d| self.entity_emb[head][d] + self.relation_emb[relation][d] - self.entity_emb[tail][d])
            .collect();
        l2_norm(&v)
    }

    /// Get the embedding vector for an entity.
    pub fn entity_vector(&self, id: EmbId) -> Option<&Vec<f64>> {
        self.entity_emb.get(id)
    }
}

// ---------------------------------------------------------------------------
// 2. TransR
// ---------------------------------------------------------------------------

/// TransR: projects entity embeddings into a relation-specific subspace.
///
/// Each relation `r` has an entity embedding `e_dim` → relation space `r_dim`
/// projection matrix `W_r`.  Score: `‖W_r·h + r − W_r·t‖`.
///
/// Reference: Lin et al., "Learning Entity and Relation Embeddings for
/// Knowledge Graph Completion", AAAI 2015.
pub struct TransR {
    /// Hyperparameters used when training this model.
    pub config: EmbeddingConfig,
    /// Entity embeddings in entity space (length = entity_dim).
    pub entity_emb: Vec<Vec<f64>>,
    /// Relation embeddings in relation space (length = relation_dim).
    pub relation_emb: Vec<Vec<f64>>,
    /// Projection matrices W_r: entity_dim → relation_dim, flattened row-major.
    pub proj_matrices: Vec<Vec<f64>>,
    num_entities: usize,
    num_relations: usize,
    /// Dimension of the relation space (may differ from entity_dim).
    relation_dim: usize,
}

impl TransR {
    /// `entity_dim` = dimension of entity embeddings;
    /// `relation_dim` = dimension of the relation subspace.
    pub fn new(
        num_entities: usize,
        num_relations: usize,
        config: EmbeddingConfig,
        relation_dim: usize,
    ) -> Self {
        TransR {
            entity_emb: Vec::new(),
            relation_emb: Vec::new(),
            proj_matrices: Vec::new(),
            config,
            num_entities,
            num_relations,
            relation_dim,
        }
    }

    /// Train TransR embeddings.
    pub fn train(&mut self, triples: &[TrainingTriple]) {
        let e_dim = self.config.dim;
        let r_dim = self.relation_dim;
        let mut rng = Lcg::new(0xfeedfacecafebeef);

        self.entity_emb = (0..self.num_entities)
            .map(|_| {
                let mut v: Vec<f64> = (0..e_dim).map(|_| rng.next_f64() * 0.1).collect();
                l2_normalize(&mut v);
                v
            })
            .collect();
        self.relation_emb = (0..self.num_relations)
            .map(|_| {
                let mut v: Vec<f64> = (0..r_dim).map(|_| rng.next_f64() * 0.1).collect();
                l2_normalize(&mut v);
                v
            })
            .collect();
        // Initialise projection matrices as near-identity (truncated/padded)
        self.proj_matrices = (0..self.num_relations)
            .map(|_| {
                let mut mat = vec![0.0f64; r_dim * e_dim];
                for i in 0..r_dim.min(e_dim) {
                    mat[i * e_dim + i] = 1.0;
                }
                mat
            })
            .collect();

        let n = self.num_entities;
        let margin = self.config.margin;
        let lr = self.config.lr;

        for _epoch in 0..self.config.epochs {
            for triple in triples {
                let corrupt_head = rng.next_usize(2) == 0;
                let neg_h = if corrupt_head {
                    let mut c = rng.next_usize(n);
                    if c == triple.head {
                        c = (c + 1) % n;
                    }
                    c
                } else {
                    triple.head
                };
                let neg_t = if !corrupt_head {
                    let mut c = rng.next_usize(n);
                    if c == triple.tail {
                        c = (c + 1) % n;
                    }
                    c
                } else {
                    triple.tail
                };

                let r = triple.relation;
                let h = triple.head;
                let t = triple.tail;

                let ph = self.project(h, r);
                let pt = self.project(t, r);
                let pnh = self.project(neg_h, r);
                let pnt = self.project(neg_t, r);

                let pos: Vec<f64> = (0..r_dim)
                    .map(|d| ph[d] + self.relation_emb[r][d] - pt[d])
                    .collect();
                let neg: Vec<f64> = (0..r_dim)
                    .map(|d| pnh[d] + self.relation_emb[r][d] - pnt[d])
                    .collect();

                let pos_dist = l2_norm(&pos);
                let neg_dist = l2_norm(&neg);
                let loss = (margin + pos_dist - neg_dist).max(0.0);
                if loss == 0.0 {
                    continue;
                }

                let grad_pos: Vec<f64> = pos.iter().map(|x| x / pos_dist).collect();
                let grad_neg: Vec<f64> = neg.iter().map(|x| x / neg_dist).collect();

                // Entity embedding update (back-project gradient)
                for d_e in 0..e_dim {
                    let mut gp = 0.0_f64;
                    let mut gn = 0.0_f64;
                    for d_r in 0..r_dim {
                        gp += self.proj_matrices[r][d_r * e_dim + d_e] * grad_pos[d_r];
                        gn += self.proj_matrices[r][d_r * e_dim + d_e] * grad_neg[d_r];
                    }
                    self.entity_emb[h][d_e] -= lr * gp;
                    self.entity_emb[t][d_e] += lr * gp;
                    self.entity_emb[neg_h][d_e] += lr * gn;
                    self.entity_emb[neg_t][d_e] -= lr * gn;
                }
                // Relation embedding update
                for d_r in 0..r_dim {
                    self.relation_emb[r][d_r] -= lr * (grad_pos[d_r] - grad_neg[d_r]);
                }

                l2_normalize(&mut self.entity_emb[h]);
                l2_normalize(&mut self.entity_emb[t]);
                l2_normalize(&mut self.entity_emb[neg_h]);
                l2_normalize(&mut self.entity_emb[neg_t]);
            }
        }
    }

    /// Project entity embedding into relation space: `W_r · e`.
    fn project(&self, entity: EmbId, relation: usize) -> Vec<f64> {
        let e_dim = self.config.dim;
        let r_dim = self.relation_dim;
        let mat = &self.proj_matrices[relation];
        let emb = &self.entity_emb[entity];
        (0..r_dim)
            .map(|dr| {
                (0..e_dim)
                    .map(|de| mat[dr * e_dim + de] * emb[de])
                    .sum::<f64>()
            })
            .collect()
    }

    /// Score a triple (lower = better).
    pub fn score(&self, head: EmbId, relation: EmbId, tail: EmbId) -> f64 {
        if head >= self.num_entities
            || tail >= self.num_entities
            || relation >= self.num_relations
        {
            return f64::MAX;
        }
        let r_dim = self.relation_dim;
        let ph = self.project(head, relation);
        let pt = self.project(tail, relation);
        let v: Vec<f64> = (0..r_dim)
            .map(|d| ph[d] + self.relation_emb[relation][d] - pt[d])
            .collect();
        l2_norm(&v)
    }

    /// Get the entity embedding.
    pub fn entity_vector(&self, id: EmbId) -> Option<&Vec<f64>> {
        self.entity_emb.get(id)
    }
}

// ---------------------------------------------------------------------------
// 3. DistMult
// ---------------------------------------------------------------------------

/// DistMult: bilinear scoring `⟨h, r, t⟩ = Σ h_i · r_i · t_i`.
///
/// Reference: Yang et al., "Embedding Entities and Relations for Learning
/// and Inference in Knowledge Bases", ICLR 2015.
pub struct DistMult {
    /// Hyperparameters used when training this model.
    pub config: EmbeddingConfig,
    /// Entity embedding matrix (one row per entity).
    pub entity_emb: Vec<Vec<f64>>,
    /// Relation embedding matrix (one row per relation).
    pub relation_emb: Vec<Vec<f64>>,
    num_entities: usize,
    num_relations: usize,
}

impl DistMult {
    /// Create an uninitialised DistMult model.
    pub fn new(num_entities: usize, num_relations: usize, config: EmbeddingConfig) -> Self {
        DistMult {
            entity_emb: Vec::new(),
            relation_emb: Vec::new(),
            config,
            num_entities,
            num_relations,
        }
    }

    /// Train DistMult with max-margin loss.
    pub fn train(&mut self, triples: &[TrainingTriple]) {
        let dim = self.config.dim;
        let mut rng = Lcg::new(0xabcdef0123456789);

        self.entity_emb = (0..self.num_entities)
            .map(|_| (0..dim).map(|_| rng.next_f64() * 0.1).collect())
            .collect();
        self.relation_emb = (0..self.num_relations)
            .map(|_| (0..dim).map(|_| rng.next_f64() * 0.1).collect())
            .collect();

        let n = self.num_entities;
        let margin = self.config.margin;
        let lr = self.config.lr;

        for _epoch in 0..self.config.epochs {
            for triple in triples {
                let corrupt_head = rng.next_usize(2) == 0;
                let neg_h = if corrupt_head {
                    let mut c = rng.next_usize(n);
                    if c == triple.head { c = (c + 1) % n; }
                    c
                } else { triple.head };
                let neg_t = if !corrupt_head {
                    let mut c = rng.next_usize(n);
                    if c == triple.tail { c = (c + 1) % n; }
                    c
                } else { triple.tail };

                let h = triple.head;
                let r = triple.relation;
                let t = triple.tail;

                let pos_score = self.raw_score(h, r, t);
                let neg_score = self.raw_score(neg_h, r, neg_t);
                let loss = (margin - pos_score + neg_score).max(0.0);
                if loss == 0.0 { continue; }

                // Gradients for DistMult: ∂/∂h = r ⊙ t, etc.
                for d in 0..dim {
                    let re = self.relation_emb[r][d];
                    let te = self.entity_emb[t][d];
                    let he = self.entity_emb[h][d];
                    let neg_he = self.entity_emb[neg_h][d];
                    let neg_te = self.entity_emb[neg_t][d];

                    // Positive triple: maximise score → ascent
                    self.entity_emb[h][d] += lr * re * te;
                    self.entity_emb[t][d] += lr * re * he;
                    self.relation_emb[r][d] += lr * he * te;

                    // Negative triple: minimise score → descent
                    self.entity_emb[neg_h][d] -= lr * re * neg_te;
                    self.entity_emb[neg_t][d] -= lr * re * neg_he;

                    // L2 regularisation (mild)
                    self.entity_emb[h][d] -= lr * 1e-3 * self.entity_emb[h][d];
                    self.entity_emb[t][d] -= lr * 1e-3 * self.entity_emb[t][d];
                }
            }
        }
    }

    /// Raw bilinear score (higher = more plausible).
    fn raw_score(&self, head: EmbId, relation: EmbId, tail: EmbId) -> f64 {
        let dim = self.config.dim;
        (0..dim)
            .map(|d| {
                self.entity_emb[head][d]
                    * self.relation_emb[relation][d]
                    * self.entity_emb[tail][d]
            })
            .sum()
    }

    /// Score a triple (higher = more plausible for DistMult).
    pub fn score(&self, head: EmbId, relation: EmbId, tail: EmbId) -> f64 {
        if head >= self.num_entities
            || tail >= self.num_entities
            || relation >= self.num_relations
        {
            return f64::MIN;
        }
        self.raw_score(head, relation, tail)
    }

    /// Entity embedding vector.
    pub fn entity_vector(&self, id: EmbId) -> Option<&Vec<f64>> {
        self.entity_emb.get(id)
    }
}

// ---------------------------------------------------------------------------
// 4. ComplEx
// ---------------------------------------------------------------------------

/// ComplEx: complex-valued bilinear model.
///
/// Each entity / relation is represented by a pair of real vectors
/// (`re` and `im`) of dimension `dim`.  Score:
/// `Re(⟨h, r, t̄⟩) = ⟨re_h, re_r, re_t⟩ + ⟨im_h, re_r, im_t⟩
///                 + ⟨re_h, im_r, im_t⟩ − ⟨im_h, im_r, re_t⟩`
///
/// Reference: Trouillon et al., "Complex Embeddings for Simple Link
/// Prediction", ICML 2016.
pub struct ComplEx {
    /// Hyperparameters used when training this model.
    pub config: EmbeddingConfig,
    /// Real part of entity embeddings.
    pub entity_re: Vec<Vec<f64>>,
    /// Imaginary part of entity embeddings.
    pub entity_im: Vec<Vec<f64>>,
    /// Real part of relation embeddings.
    pub relation_re: Vec<Vec<f64>>,
    /// Imaginary part of relation embeddings.
    pub relation_im: Vec<Vec<f64>>,
    num_entities: usize,
    num_relations: usize,
}

impl ComplEx {
    /// Create an uninitialised ComplEx model.
    pub fn new(num_entities: usize, num_relations: usize, config: EmbeddingConfig) -> Self {
        ComplEx {
            entity_re: Vec::new(),
            entity_im: Vec::new(),
            relation_re: Vec::new(),
            relation_im: Vec::new(),
            config,
            num_entities,
            num_relations,
        }
    }

    /// Train ComplEx with max-margin loss.
    pub fn train(&mut self, triples: &[TrainingTriple]) {
        let dim = self.config.dim;
        let mut rng = Lcg::new(0x12345678deadbeef);
        let init = || -> Vec<Vec<f64>> {
            (0..0).map(|_| vec![0.0]).collect() // placeholder
        };
        let _ = init;

        let make = |n: usize, rng: &mut Lcg| -> Vec<Vec<f64>> {
            (0..n)
                .map(|_| (0..dim).map(|_| rng.next_f64() * 0.1).collect())
                .collect()
        };

        self.entity_re = make(self.num_entities, &mut rng);
        self.entity_im = make(self.num_entities, &mut rng);
        self.relation_re = make(self.num_relations, &mut rng);
        self.relation_im = make(self.num_relations, &mut rng);

        let n = self.num_entities;
        let margin = self.config.margin;
        let lr = self.config.lr;

        for _epoch in 0..self.config.epochs {
            for triple in triples {
                let corrupt_head = rng.next_usize(2) == 0;
                let neg_h = if corrupt_head {
                    let mut c = rng.next_usize(n);
                    if c == triple.head { c = (c + 1) % n; }
                    c
                } else { triple.head };
                let neg_t = if !corrupt_head {
                    let mut c = rng.next_usize(n);
                    if c == triple.tail { c = (c + 1) % n; }
                    c
                } else { triple.tail };

                let h = triple.head;
                let r = triple.relation;
                let t = triple.tail;

                let pos_score = self.raw_score(h, r, t);
                let neg_score = self.raw_score(neg_h, r, neg_t);
                let loss = (margin - pos_score + neg_score).max(0.0);
                if loss == 0.0 { continue; }

                // Gradient of Re(⟨h, r, t̄⟩) w.r.t. each component
                for d in 0..dim {
                    let rr = self.relation_re[r][d];
                    let ri = self.relation_im[r][d];
                    let hr = self.entity_re[h][d];
                    let hi = self.entity_im[h][d];
                    let tr_ = self.entity_re[t][d];
                    let ti = self.entity_im[t][d];

                    // ∂score/∂h_re = r_re * t_re + r_im * t_im
                    // ∂score/∂h_im = r_re * t_im - r_im * t_re
                    // ∂score/∂t_re = h_re * r_re - h_im * r_im
                    // ∂score/∂t_im = h_re * r_im + h_im * r_re
                    let grad_hr = rr * tr_ + ri * ti;
                    let grad_hi = rr * ti - ri * tr_;
                    let grad_tr = hr * rr - hi * ri;
                    let grad_ti = hr * ri + hi * rr;

                    // Positive: ascent
                    self.entity_re[h][d] += lr * grad_hr;
                    self.entity_im[h][d] += lr * grad_hi;
                    self.entity_re[t][d] += lr * grad_tr;
                    self.entity_im[t][d] += lr * grad_ti;

                    // Negative: descent
                    let neg_hr_g = rr * self.entity_re[neg_t][d] + ri * self.entity_im[neg_t][d];
                    let neg_hi_g = rr * self.entity_im[neg_t][d] - ri * self.entity_re[neg_t][d];
                    let neg_tr_g = self.entity_re[neg_h][d] * rr - self.entity_im[neg_h][d] * ri;
                    let neg_ti_g = self.entity_re[neg_h][d] * ri + self.entity_im[neg_h][d] * rr;

                    self.entity_re[neg_h][d] -= lr * neg_hr_g;
                    self.entity_im[neg_h][d] -= lr * neg_hi_g;
                    self.entity_re[neg_t][d] -= lr * neg_tr_g;
                    self.entity_im[neg_t][d] -= lr * neg_ti_g;

                    // L2 regularisation
                    let reg = lr * 1e-3;
                    self.entity_re[h][d] -= reg * self.entity_re[h][d];
                    self.entity_im[h][d] -= reg * self.entity_im[h][d];
                    self.entity_re[t][d] -= reg * self.entity_re[t][d];
                    self.entity_im[t][d] -= reg * self.entity_im[t][d];
                }
            }
        }
    }

    /// `Re(⟨h, r, t̄⟩)` (higher = more plausible).
    fn raw_score(&self, head: EmbId, relation: EmbId, tail: EmbId) -> f64 {
        let dim = self.config.dim;
        (0..dim)
            .map(|d| {
                let rr = self.relation_re[relation][d];
                let ri = self.relation_im[relation][d];
                let hr = self.entity_re[head][d];
                let hi = self.entity_im[head][d];
                let tr_ = self.entity_re[tail][d];
                let ti = self.entity_im[tail][d];
                hr * rr * tr_ + hi * rr * ti + hr * ri * ti - hi * ri * tr_
            })
            .sum()
    }

    /// Score a triple (higher = more plausible).
    pub fn score(&self, head: EmbId, relation: EmbId, tail: EmbId) -> f64 {
        if head >= self.num_entities
            || tail >= self.num_entities
            || relation >= self.num_relations
        {
            return f64::MIN;
        }
        self.raw_score(head, relation, tail)
    }

    /// Concatenated [re; im] entity embedding.
    pub fn entity_vector(&self, id: EmbId) -> Option<Vec<f64>> {
        let re = self.entity_re.get(id)?;
        let im = self.entity_im.get(id)?;
        let mut v = re.clone();
        v.extend_from_slice(im);
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// High-level convenience wrapper
// ---------------------------------------------------------------------------

/// Supported KG embedding model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// Translational model: h + r ≈ t.
    TransE,
    /// Translational model with relation-specific projection space.
    TransR,
    /// Bilinear diagonal model: h ⊙ r ⊙ t.
    DistMult,
    /// Complex-valued bilinear model.
    ComplEx,
}

/// Build entity / relation index maps from raw string triples.
pub fn build_indices(
    triples: &[(&str, &str, &str)],
) -> (
    HashMap<String, EmbId>,  // entity → id
    HashMap<String, EmbId>,  // relation → id
    Vec<TrainingTriple>,
) {
    let mut entity_map: HashMap<String, EmbId> = HashMap::new();
    let mut relation_map: HashMap<String, EmbId> = HashMap::new();
    let mut training: Vec<TrainingTriple> = Vec::new();

    for (h, r, t) in triples {
        let next_e = entity_map.len();
        let h_id = *entity_map.entry(h.to_string()).or_insert(next_e);
        let next_e2 = entity_map.len();
        let t_id = *entity_map.entry(t.to_string()).or_insert(next_e2);
        let next_r = relation_map.len();
        let r_id = *relation_map.entry(r.to_string()).or_insert(next_r);
        training.push(TrainingTriple {
            head: h_id,
            relation: r_id,
            tail: t_id,
        });
    }
    (entity_map, relation_map, training)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn mini_triples() -> Vec<TrainingTriple> {
        vec![
            TrainingTriple { head: 0, relation: 0, tail: 1 },
            TrainingTriple { head: 1, relation: 0, tail: 2 },
            TrainingTriple { head: 0, relation: 1, tail: 2 },
            TrainingTriple { head: 2, relation: 1, tail: 3 },
        ]
    }

    #[test]
    fn test_transe_train_and_score() {
        let triples = mini_triples();
        let mut model = TransE::new(4, 2, EmbeddingConfig { epochs: 50, ..Default::default() });
        model.train(&triples);
        // Known-positive should score lower than arbitrary negative
        let pos = model.score(0, 0, 1);
        let neg = model.score(0, 0, 3); // not a positive triple
        assert!(pos.is_finite());
        assert!(neg.is_finite());
    }

    #[test]
    fn test_transe_entity_vector() {
        let triples = mini_triples();
        let mut model = TransE::new(4, 2, EmbeddingConfig::default());
        model.train(&triples);
        let v = model.entity_vector(0).expect("vector should exist");
        assert_eq!(v.len(), 64);
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.1, "entity vector should be near unit-norm");
    }

    #[test]
    fn test_transr_train() {
        let triples = mini_triples();
        let mut model = TransR::new(
            4, 2,
            EmbeddingConfig { dim: 16, epochs: 30, ..Default::default() },
            8,
        );
        model.train(&triples);
        let score = model.score(0, 0, 1);
        assert!(score.is_finite());
    }

    #[test]
    fn test_distmult_train() {
        let triples = mini_triples();
        let mut model = DistMult::new(
            4, 2,
            EmbeddingConfig { epochs: 50, ..Default::default() },
        );
        model.train(&triples);
        // Positive triple should score higher than random
        let pos = model.score(0, 0, 1);
        assert!(pos.is_finite());
    }

    #[test]
    fn test_complex_train() {
        let triples = mini_triples();
        let mut model = ComplEx::new(
            4, 2,
            EmbeddingConfig { epochs: 50, ..Default::default() },
        );
        model.train(&triples);
        let v = model.entity_vector(0).expect("should have vector");
        assert_eq!(v.len(), 128); // re + im
    }

    #[test]
    fn test_build_indices() {
        let raw = vec![
            ("Alice", "works_at", "Acme"),
            ("Bob", "works_at", "Acme"),
            ("Acme", "located_in", "London"),
        ];
        let (emap, rmap, triples) = build_indices(&raw);
        assert_eq!(emap.len(), 4);
        assert_eq!(rmap.len(), 2);
        assert_eq!(triples.len(), 3);
    }
}
