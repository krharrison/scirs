//! Recommender-system dataset generators.
//!
//! This module provides synthetic datasets for collaborative filtering,
//! content-based filtering, and hybrid recommender systems.
//!
//! All generators are self-contained (Park-Miller LCG, no `rand` crate).

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Internal LCG RNG
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
    /// N(0, 1) via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InteractionMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse user-item interaction matrix stored in coordinate (COO) format.
///
/// Interactions are stored as `(user_id, item_id, rating)` triples.
/// Optional timestamps are supported for session-aware models.
#[derive(Debug, Clone)]
pub struct InteractionMatrix {
    /// Number of unique users.
    pub n_users: usize,
    /// Number of unique items.
    pub n_items: usize,
    /// User identifier for each observed interaction.
    pub user_ids: Vec<usize>,
    /// Item identifier for each observed interaction.
    pub item_ids: Vec<usize>,
    /// Rating value for each interaction.
    pub ratings: Vec<f64>,
    /// Optional Unix-timestamp for each interaction.
    pub timestamps: Option<Vec<u64>>,
}

impl InteractionMatrix {
    /// Create an empty interaction matrix.
    pub fn new(n_users: usize, n_items: usize) -> Self {
        Self {
            n_users,
            n_items,
            user_ids: Vec::new(),
            item_ids: Vec::new(),
            ratings: Vec::new(),
            timestamps: None,
        }
    }

    /// Append one interaction (unchecked bounds for performance; validate inputs yourself).
    pub fn add_interaction(&mut self, user: usize, item: usize, rating: f64) {
        self.user_ids.push(user);
        self.item_ids.push(item);
        self.ratings.push(rating);
        if let Some(ref mut ts) = self.timestamps {
            ts.push(0);
        }
    }

    /// Number of recorded interactions.
    pub fn n_interactions(&self) -> usize {
        self.ratings.len()
    }

    /// Convert to a dense `n_users × n_items` matrix (ratings not seen → `0.0`).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix would exceed 1 GiB.
    pub fn to_dense(&self) -> Result<Vec<Vec<f64>>> {
        let n_cells = self.n_users.saturating_mul(self.n_items);
        if n_cells > 134_217_728 {
            // 128 M cells × 8 B = 1 GiB
            return Err(DatasetsError::InvalidFormat(
                "to_dense: matrix too large (> 1 GiB); use sparse representation instead"
                    .to_string(),
            ));
        }
        let mut dense = vec![vec![0.0f64; self.n_items]; self.n_users];
        for ((u, i), r) in self
            .user_ids
            .iter()
            .zip(self.item_ids.iter())
            .zip(self.ratings.iter())
        {
            if *u < self.n_users && *i < self.n_items {
                dense[*u][*i] = *r;
            }
        }
        Ok(dense)
    }

    /// Fraction of user-item pairs that are observed.
    pub fn density(&self) -> f64 {
        let total = self.n_users * self.n_items;
        if total == 0 {
            return 0.0;
        }
        self.ratings.len() as f64 / total as f64
    }

    /// Mean rating over all observed interactions.
    pub fn mean_rating(&self) -> f64 {
        if self.ratings.is_empty() {
            return 0.0;
        }
        self.ratings.iter().sum::<f64>() / self.ratings.len() as f64
    }

    /// Split interactions into train / test sets by randomly holding out
    /// `test_ratio` of each user's interactions.
    ///
    /// # Arguments
    ///
    /// * `test_ratio` – Fraction of interactions per user held out (0 < ratio < 1).
    /// * `seed`       – Reproducibility seed.
    ///
    /// # Errors
    ///
    /// Returns an error if `test_ratio` is not in `(0, 1)`.
    pub fn user_split(&self, test_ratio: f64, seed: u64) -> Result<(Self, Self)> {
        if !(0.0 < test_ratio && test_ratio < 1.0) {
            return Err(DatasetsError::InvalidFormat(
                "user_split: test_ratio must be in (0, 1)".to_string(),
            ));
        }

        let mut rng = Lcg::new(seed);

        // Group interaction indices by user.
        let mut user_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n_users];
        for (idx, &u) in self.user_ids.iter().enumerate() {
            if u < self.n_users {
                user_indices[u].push(idx);
            }
        }

        let mut train = Self::new(self.n_users, self.n_items);
        let mut test = Self::new(self.n_users, self.n_items);
        if self.timestamps.is_some() {
            train.timestamps = Some(Vec::new());
            test.timestamps = Some(Vec::new());
        }

        for indices in &user_indices {
            let n_test = ((indices.len() as f64 * test_ratio).round() as usize).max(0);
            // Randomly tag n_test indices as test.
            let mut is_test = vec![false; indices.len()];
            // Fisher-Yates partial shuffle to pick n_test positions.
            let mut perm: Vec<usize> = (0..indices.len()).collect();
            for k in 0..n_test {
                let j = k + rng.next_usize(indices.len() - k);
                perm.swap(k, j);
            }
            for k in 0..n_test {
                is_test[perm[k]] = true;
            }

            for (pos, &global_idx) in indices.iter().enumerate() {
                let u = self.user_ids[global_idx];
                let i = self.item_ids[global_idx];
                let r = self.ratings[global_idx];
                if is_test[pos] {
                    test.user_ids.push(u);
                    test.item_ids.push(i);
                    test.ratings.push(r);
                    if let (Some(ref ts_src), Some(ref mut ts_dst)) =
                        (&self.timestamps, &mut test.timestamps)
                    {
                        ts_dst.push(ts_src[global_idx]);
                    }
                } else {
                    train.user_ids.push(u);
                    train.item_ids.push(i);
                    train.ratings.push(r);
                    if let (Some(ref ts_src), Some(ref mut ts_dst)) =
                        (&self.timestamps, &mut train.timestamps)
                    {
                        ts_dst.push(ts_src[global_idx]);
                    }
                }
            }
        }

        Ok((train, test))
    }

    /// Leave-one-out split: the last interaction per user (by list order) goes to test.
    ///
    /// Users with fewer than 2 interactions keep all interactions in train and
    /// contribute nothing to test.
    pub fn leave_one_out(&self) -> (Self, Self) {
        // Find last index per user.
        let mut last_idx: Vec<Option<usize>> = vec![None; self.n_users];
        for (idx, &u) in self.user_ids.iter().enumerate() {
            if u < self.n_users {
                last_idx[u] = Some(idx);
            }
        }

        let test_set: std::collections::HashSet<usize> =
            last_idx.iter().filter_map(|opt| *opt).collect();

        let mut train = Self::new(self.n_users, self.n_items);
        let mut test = Self::new(self.n_users, self.n_items);
        if self.timestamps.is_some() {
            train.timestamps = Some(Vec::new());
            test.timestamps = Some(Vec::new());
        }

        for (idx, ((&u, &i), &r)) in self
            .user_ids
            .iter()
            .zip(self.item_ids.iter())
            .zip(self.ratings.iter())
            .enumerate()
        {
            if test_set.contains(&idx) {
                test.user_ids.push(u);
                test.item_ids.push(i);
                test.ratings.push(r);
                if let (Some(ref ts_src), Some(ref mut ts_dst)) =
                    (&self.timestamps, &mut test.timestamps)
                {
                    ts_dst.push(ts_src[idx]);
                }
            } else {
                train.user_ids.push(u);
                train.item_ids.push(i);
                train.ratings.push(r);
                if let (Some(ref ts_src), Some(ref mut ts_dst)) =
                    (&self.timestamps, &mut train.timestamps)
                {
                    ts_dst.push(ts_src[idx]);
                }
            }
        }

        (train, test)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// make_rating_dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic collaborative-filtering dataset via latent-factor model.
///
/// Ratings are computed as `clip(U[u] · V[i] + noise, 1.0, 5.0)` where `U` and
/// `V` are random latent factor matrices.  Only a `density` fraction of all
/// user-item pairs are observed (sampled uniformly at random).
///
/// # Arguments
///
/// * `n_users`         – Number of users.
/// * `n_items`         – Number of items.
/// * `density`         – Fraction of user-item pairs observed (0 < density ≤ 1).
/// * `n_latent_factors`– Dimension of the latent space.
/// * `noise_std`       – Standard deviation of Gaussian noise added to ratings.
/// * `seed`            – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if any argument is out of range.
pub fn make_rating_dataset(
    n_users: usize,
    n_items: usize,
    density: f64,
    n_latent_factors: usize,
    noise_std: f64,
    seed: u64,
) -> Result<InteractionMatrix> {
    if n_users == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_rating_dataset: n_users must be >= 1".to_string(),
        ));
    }
    if n_items == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_rating_dataset: n_items must be >= 1".to_string(),
        ));
    }
    if !(density > 0.0 && density <= 1.0) {
        return Err(DatasetsError::InvalidFormat(
            "make_rating_dataset: density must be in (0, 1]".to_string(),
        ));
    }
    if n_latent_factors == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_rating_dataset: n_latent_factors must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);

    // Build latent factor matrices U (n_users × k) and V (n_items × k).
    let k = n_latent_factors;
    let scale = 1.0 / (k as f64).sqrt();

    let user_factors: Vec<Vec<f64>> = (0..n_users)
        .map(|_| (0..k).map(|_| rng.next_normal() * scale).collect())
        .collect();
    let item_factors: Vec<Vec<f64>> = (0..n_items)
        .map(|_| (0..k).map(|_| rng.next_normal() * scale).collect())
        .collect();

    let n_target = ((n_users * n_items) as f64 * density).round() as usize;

    let mut matrix = InteractionMatrix::new(n_users, n_items);
    // Track which (user, item) pairs have been added (avoid duplicates).
    let mut seen: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::with_capacity(n_target);

    let max_attempts = (n_target * 8).max(1);
    let mut attempts = 0usize;

    while matrix.n_interactions() < n_target && attempts < max_attempts {
        attempts += 1;
        let u = rng.next_usize(n_users);
        let i = rng.next_usize(n_items);

        if seen.contains(&(u, i)) {
            continue;
        }
        seen.insert((u, i));

        // Dot product of latent factors.
        let dot: f64 = user_factors[u]
            .iter()
            .zip(item_factors[i].iter())
            .map(|(a, b)| a * b)
            .sum();

        // Shift to [1, 5] range and add noise.
        let raw = dot * 2.5 + 3.0 + rng.next_normal() * noise_std;
        let rating = raw.clamp(1.0, 5.0);

        matrix.user_ids.push(u);
        matrix.item_ids.push(i);
        matrix.ratings.push(rating);
    }

    Ok(matrix)
}

// ─────────────────────────────────────────────────────────────────────────────
// MovieDataset
// ─────────────────────────────────────────────────────────────────────────────

/// MovieLens-style dataset with genre features and synthetic user demographics.
#[derive(Debug, Clone)]
pub struct MovieDataset {
    /// User-item rating interactions.
    pub ratings: InteractionMatrix,
    /// Item (movie) content features: one row per movie, one column per genre (binary).
    pub item_features: Vec<Vec<f64>>,
    /// User demographic features: one row per user (age_norm, gender, activity_level).
    pub user_features: Vec<Vec<f64>>,
    /// Synthetic movie titles.
    pub item_names: Vec<String>,
    /// Feature column names (genres).
    pub feature_names: Vec<String>,
}

static GENRE_NAMES: &[&str] = &[
    "Action",
    "Comedy",
    "Drama",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Animation",
    "Documentary",
    "Fantasy",
    "Mystery",
    "Adventure",
    "Musical",
    "Western",
    "Crime",
    "Biography",
];

static MOVIE_ADJECTIVES: &[&str] = &[
    "Dark", "Bright", "Lost", "Hidden", "Final", "Rising", "Last", "First", "Great", "Little",
    "Silent", "Golden", "Shadow", "Eternal", "Fallen",
];

static MOVIE_NOUNS: &[&str] = &[
    "Storm", "Dawn", "Night", "Dream", "Fire", "Star", "Light", "World", "Road", "Heart", "Time",
    "Sky", "Rain", "Wind", "Horizon",
];

fn make_movie_name(idx: usize, rng: &mut Lcg) -> String {
    let adj = MOVIE_ADJECTIVES[rng.next_usize(MOVIE_ADJECTIVES.len())];
    let noun = MOVIE_NOUNS[rng.next_usize(MOVIE_NOUNS.len())];
    format!("The {adj} {noun} {}", idx + 1)
}

/// Generate a MovieLens-style dataset with content features and demographics.
///
/// # Arguments
///
/// * `n_users`  – Number of users.
/// * `n_movies` – Number of movies.
/// * `n_genres` – Number of genre dimensions (≤ 16; capped internally).
/// * `density`  – Fraction of user-movie pairs rated (0 < density ≤ 1).
/// * `seed`     – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if arguments are out of range.
pub fn make_movie_dataset(
    n_users: usize,
    n_movies: usize,
    n_genres: usize,
    density: f64,
    seed: u64,
) -> Result<MovieDataset> {
    if n_users == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_movie_dataset: n_users must be >= 1".to_string(),
        ));
    }
    if n_movies == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_movie_dataset: n_movies must be >= 1".to_string(),
        ));
    }
    let n_genres = n_genres.min(GENRE_NAMES.len()).max(1);
    if !(density > 0.0 && density <= 1.0) {
        return Err(DatasetsError::InvalidFormat(
            "make_movie_dataset: density must be in (0, 1]".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);

    // Generate item (movie) genre features: each movie has 1-3 genres set.
    let feature_names: Vec<String> = GENRE_NAMES[..n_genres]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let item_features: Vec<Vec<f64>> = (0..n_movies)
        .map(|_| {
            let mut feat = vec![0.0f64; n_genres];
            let n_active = 1 + rng.next_usize(3.min(n_genres));
            // Pick n_active distinct genres.
            let mut indices: Vec<usize> = (0..n_genres).collect();
            for k in 0..n_active {
                let j = k + rng.next_usize(n_genres - k);
                indices.swap(k, j);
            }
            for k in 0..n_active {
                feat[indices[k]] = 1.0;
            }
            feat
        })
        .collect();

    // Generate user demographic features: (age_norm, gender_binary, activity_level).
    // age_norm ~ U(0,1), gender ~ Bernoulli(0.5), activity ~ U(0,1).
    let user_features: Vec<Vec<f64>> = (0..n_users)
        .map(|_| {
            vec![
                rng.next_f64(),                               // age_norm
                if rng.next_f64() < 0.5 { 0.0 } else { 1.0 }, // gender
                rng.next_f64(),                               // activity_level
            ]
        })
        .collect();

    // Movie names.
    let item_names: Vec<String> = (0..n_movies)
        .map(|idx| make_movie_name(idx, &mut rng))
        .collect();

    // Build ratings via latent factor model (n_latent = n_genres as proxy).
    let n_latent = n_genres;
    // User latent vector derived from user_features (padded/truncated to n_latent).
    // Item latent vector = genre feature vector (already length n_genres).
    // This ensures genre-preferences are meaningful.
    let user_latent: Vec<Vec<f64>> = user_features
        .iter()
        .map(|uf| {
            // Tile/extend user feature to n_latent dimensions.
            let mut v = uf.clone();
            while v.len() < n_latent {
                v.push(rng.next_f64() * 0.5);
            }
            v.truncate(n_latent);
            v
        })
        .collect();

    let n_target = ((n_users * n_movies) as f64 * density).round() as usize;
    let mut ratings = InteractionMatrix::new(n_users, n_movies);
    let mut seen: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::with_capacity(n_target);

    let max_attempts = (n_target * 8).max(1);
    let mut attempts = 0usize;

    while ratings.n_interactions() < n_target && attempts < max_attempts {
        attempts += 1;
        let u = rng.next_usize(n_users);
        let m = rng.next_usize(n_movies);

        if seen.contains(&(u, m)) {
            continue;
        }
        seen.insert((u, m));

        let dot: f64 = user_latent[u]
            .iter()
            .zip(item_features[m].iter())
            .map(|(a, b)| a * b)
            .sum();

        let raw = dot * 2.5 + 3.0 + rng.next_normal() * 0.5;
        let rating = raw.clamp(1.0, 5.0);

        ratings.user_ids.push(u);
        ratings.item_ids.push(m);
        ratings.ratings.push(rating);
    }

    Ok(MovieDataset {
        ratings,
        item_features,
        user_features,
        item_names,
        feature_names,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interaction_matrix_basic() {
        let mut m = InteractionMatrix::new(3, 4);
        m.add_interaction(0, 1, 4.5);
        m.add_interaction(1, 3, 2.0);
        assert_eq!(m.n_interactions(), 2);
        assert!((m.mean_rating() - 3.25).abs() < 1e-9);
        assert!((m.density() - 2.0 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_to_dense() {
        let mut m = InteractionMatrix::new(2, 3);
        m.add_interaction(0, 2, 3.0);
        m.add_interaction(1, 0, 5.0);
        let d = m.to_dense().expect("to_dense failed");
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].len(), 3);
        assert!((d[0][2] - 3.0).abs() < 1e-9);
        assert!((d[1][0] - 5.0).abs() < 1e-9);
        assert_eq!(d[0][0], 0.0);
    }

    #[test]
    fn test_user_split() {
        let ds = make_rating_dataset(20, 30, 0.3, 5, 0.5, 42).expect("rating dataset");
        let total = ds.n_interactions();
        let (train, test) = ds.user_split(0.2, 7).expect("user_split");
        assert_eq!(train.n_interactions() + test.n_interactions(), total);
        assert!(test.n_interactions() > 0);
    }

    #[test]
    fn test_leave_one_out() {
        let ds = make_rating_dataset(10, 20, 0.4, 3, 0.3, 99).expect("rating dataset");
        let total = ds.n_interactions();
        let (train, test) = ds.leave_one_out();
        assert_eq!(train.n_interactions() + test.n_interactions(), total);
        // At most n_users interactions can be in test.
        assert!(test.n_interactions() <= 10);
    }

    #[test]
    fn test_make_rating_dataset() {
        let ds = make_rating_dataset(50, 100, 0.05, 8, 0.3, 42).expect("rating dataset");
        assert_eq!(ds.n_users, 50);
        assert_eq!(ds.n_items, 100);
        assert!(!ds.ratings.is_empty());
        for &r in &ds.ratings {
            assert!(r >= 1.0 && r <= 5.0, "rating out of [1,5]: {r}");
        }
    }

    #[test]
    fn test_make_rating_dataset_errors() {
        assert!(make_rating_dataset(0, 10, 0.1, 3, 0.1, 1).is_err());
        assert!(make_rating_dataset(5, 0, 0.1, 3, 0.1, 1).is_err());
        assert!(make_rating_dataset(5, 10, 0.0, 3, 0.1, 1).is_err());
        assert!(make_rating_dataset(5, 10, 0.1, 0, 0.1, 1).is_err());
    }

    #[test]
    fn test_make_movie_dataset() {
        let ds = make_movie_dataset(30, 50, 5, 0.1, 77).expect("movie dataset");
        assert_eq!(ds.ratings.n_users, 30);
        assert_eq!(ds.ratings.n_items, 50);
        assert_eq!(ds.item_features.len(), 50);
        assert_eq!(ds.user_features.len(), 30);
        assert_eq!(ds.item_names.len(), 50);
        assert_eq!(ds.feature_names.len(), 5);
        for feat in &ds.item_features {
            assert_eq!(feat.len(), 5);
        }
        for uf in &ds.user_features {
            assert_eq!(uf.len(), 3);
        }
    }

    #[test]
    fn test_rating_reproducibility() {
        let a = make_rating_dataset(10, 20, 0.2, 4, 0.2, 55).expect("a");
        let b = make_rating_dataset(10, 20, 0.2, 4, 0.2, 55).expect("b");
        assert_eq!(a.ratings, b.ratings);
        assert_eq!(a.user_ids, b.user_ids);
    }
}
