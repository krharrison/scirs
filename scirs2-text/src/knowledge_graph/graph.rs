//! Knowledge Graph data structure and core operations
//!
//! Provides a typed knowledge graph with entity management, relation triples,
//! graph traversal algorithms, and TransE-style entity embeddings.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{Result, TextError};

/// Unique identifier for an entity in the knowledge graph.
pub type EntityId = usize;

/// A relation triple: (subject, predicate, object) with a confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    /// The subject entity identifier.
    pub subject: EntityId,
    /// The predicate (relation type) as a string label.
    pub predicate: String,
    /// The object entity identifier.
    pub object: EntityId,
    /// Confidence score in [0, 1].
    pub confidence: f64,
}

/// A knowledge graph consisting of typed entities and directed, labeled relations.
///
/// # Example
/// ```rust
/// use scirs2_text::knowledge_graph::KnowledgeGraph;
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_entity("Alice", "Person");
/// kg.add_entity("Acme", "Organization");
/// kg.add_relation("Alice", "works_at", "Acme", 0.95);
///
/// let triples = kg.query_relations("Alice");
/// assert_eq!(triples.len(), 1);
/// assert_eq!(triples[0].predicate, "works_at");
/// ```
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// Mapping from canonical entity name to its numeric id.
    entities: HashMap<String, EntityId>,
    /// Reverse mapping from id to canonical name.
    id_to_name: Vec<String>,
    /// All relation triples stored in the graph.
    relations: Vec<Triple>,
    /// Per-entity type labels (an entity may have multiple types).
    entity_types: HashMap<EntityId, Vec<String>>,
    /// Per-entity arbitrary key/value properties.
    properties: HashMap<EntityId, HashMap<String, String>>,
    /// TransE-style embedding vectors (dimension = `embedding_dim`).
    embeddings: HashMap<EntityId, Vec<f64>>,
    /// Dimensionality used when embeddings are (re-)computed.
    embedding_dim: usize,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph with the default embedding dimension (64).
    pub fn new() -> Self {
        KnowledgeGraph {
            entities: HashMap::new(),
            id_to_name: Vec::new(),
            relations: Vec::new(),
            entity_types: HashMap::new(),
            properties: HashMap::new(),
            embeddings: HashMap::new(),
            embedding_dim: 64,
        }
    }

    /// Create an empty knowledge graph with a custom embedding dimension.
    pub fn with_embedding_dim(embedding_dim: usize) -> Self {
        KnowledgeGraph {
            embedding_dim,
            ..Self::new()
        }
    }

    // -----------------------------------------------------------------------
    // Entity management
    // -----------------------------------------------------------------------

    /// Add (or look up) an entity by name and assign the given type.
    ///
    /// If the entity already exists its id is returned and the type is
    /// appended if not already present.
    pub fn add_entity(&mut self, name: &str, entity_type: &str) -> EntityId {
        if let Some(&id) = self.entities.get(name) {
            let types = self.entity_types.entry(id).or_default();
            if !types.contains(&entity_type.to_string()) {
                types.push(entity_type.to_string());
            }
            return id;
        }
        let id = self.id_to_name.len();
        self.entities.insert(name.to_string(), id);
        self.id_to_name.push(name.to_string());
        self.entity_types
            .entry(id)
            .or_default()
            .push(entity_type.to_string());
        id
    }

    /// Look up an entity id by name.  Returns `None` when unknown.
    pub fn entity_id(&self, name: &str) -> Option<EntityId> {
        self.entities.get(name).copied()
    }

    /// Return the canonical name for an entity id.
    pub fn entity_name(&self, id: EntityId) -> Option<&str> {
        self.id_to_name.get(id).map(|s| s.as_str())
    }

    /// Return all entity names registered in the graph.
    pub fn entities(&self) -> impl Iterator<Item = &str> {
        self.id_to_name.iter().map(|s| s.as_str())
    }

    /// Number of entities.
    pub fn num_entities(&self) -> usize {
        self.id_to_name.len()
    }

    /// Number of relation triples.
    pub fn num_triples(&self) -> usize {
        self.relations.len()
    }

    /// Return the type labels of an entity.
    pub fn entity_types(&self, name: &str) -> Vec<&str> {
        match self.entities.get(name) {
            None => Vec::new(),
            Some(&id) => self
                .entity_types
                .get(&id)
                .map(|v| v.iter().map(|s| s.as_str()).collect())
                .unwrap_or_default(),
        }
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    /// Set an arbitrary key/value property on a named entity.
    pub fn set_property(&mut self, entity: &str, key: &str, value: &str) -> Result<()> {
        let id = self
            .entities
            .get(entity)
            .copied()
            .ok_or_else(|| TextError::InvalidInput(format!("Unknown entity: {entity}")))?;
        self.properties
            .entry(id)
            .or_default()
            .insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Retrieve a property value.
    pub fn get_property(&self, entity: &str, key: &str) -> Option<&str> {
        let id = self.entities.get(entity)?;
        self.properties
            .get(id)?
            .get(key)
            .map(|s| s.as_str())
    }

    // -----------------------------------------------------------------------
    // Relation management
    // -----------------------------------------------------------------------

    /// Add a relation triple.  Both entities are auto-created if absent
    /// (using the placeholder type "Unknown").
    pub fn add_relation(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
    ) {
        let s_id = if let Some(&id) = self.entities.get(subject) {
            id
        } else {
            self.add_entity(subject, "Unknown")
        };
        let o_id = if let Some(&id) = self.entities.get(object) {
            id
        } else {
            self.add_entity(object, "Unknown")
        };
        self.relations.push(Triple {
            subject: s_id,
            predicate: predicate.to_string(),
            object: o_id,
            confidence,
        });
    }

    /// Return all triples whose subject matches `subject`.
    pub fn query_relations(&self, subject: &str) -> Vec<&Triple> {
        match self.entities.get(subject) {
            None => Vec::new(),
            Some(&id) => self
                .relations
                .iter()
                .filter(|t| t.subject == id)
                .collect(),
        }
    }

    /// Return all triples whose object matches `object`.
    pub fn query_incoming(&self, object: &str) -> Vec<&Triple> {
        match self.entities.get(object) {
            None => Vec::new(),
            Some(&id) => self
                .relations
                .iter()
                .filter(|t| t.object == id)
                .collect(),
        }
    }

    /// Return all triples that involve `entity` as subject or object.
    pub fn query_all(&self, entity: &str) -> Vec<&Triple> {
        match self.entities.get(entity) {
            None => Vec::new(),
            Some(&id) => self
                .relations
                .iter()
                .filter(|t| t.subject == id || t.object == id)
                .collect(),
        }
    }

    /// Return all triples with a given predicate label.
    pub fn query_by_predicate(&self, predicate: &str) -> Vec<&Triple> {
        self.relations
            .iter()
            .filter(|t| t.predicate == predicate)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Graph algorithms
    // -----------------------------------------------------------------------

    /// BFS shortest path between two named entities.
    ///
    /// Returns `Some(path)` where `path` is the sequence of entity names from
    /// `from` to `to` (inclusive), or `None` when no path exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let &from_id = self.entities.get(from)?;
        let &to_id = self.entities.get(to)?;

        if from_id == to_id {
            return Some(vec![from.to_string()]);
        }

        // Build adjacency list (undirected for reachability)
        let mut adj: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
        for triple in &self.relations {
            adj.entry(triple.subject)
                .or_default()
                .push(triple.object);
            adj.entry(triple.object)
                .or_default()
                .push(triple.subject);
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<(EntityId, Vec<EntityId>)> = VecDeque::new();
        visited.insert(from_id);
        queue.push_back((from_id, vec![from_id]));

        while let Some((current, path)) = queue.pop_front() {
            if let Some(neighbors) = adj.get(&current) {
                for &next in neighbors {
                    if visited.contains(&next) {
                        continue;
                    }
                    let mut new_path = path.clone();
                    new_path.push(next);
                    if next == to_id {
                        return Some(
                            new_path
                                .iter()
                                .map(|&id| {
                                    self.id_to_name
                                        .get(id)
                                        .cloned()
                                        .unwrap_or_else(|| id.to_string())
                                })
                                .collect(),
                        );
                    }
                    visited.insert(next);
                    queue.push_back((next, new_path));
                }
            }
        }
        None
    }

    /// Extract a sub-graph centered on `center` up to `depth` hops away.
    pub fn subgraph(&self, center: &str, depth: usize) -> KnowledgeGraph {
        let Some(&center_id) = self.entities.get(center) else {
            return KnowledgeGraph::new();
        };

        // BFS to collect reachable entity ids within `depth` hops
        let mut reachable: HashSet<EntityId> = HashSet::new();
        reachable.insert(center_id);
        let mut frontier: HashSet<EntityId> = [center_id].into();

        // Build adjacency list
        let mut adj: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
        for t in &self.relations {
            adj.entry(t.subject).or_default().push(t.object);
            adj.entry(t.object).or_default().push(t.subject);
        }

        for _ in 0..depth {
            let mut next_frontier: HashSet<EntityId> = HashSet::new();
            for &node in &frontier {
                if let Some(neighbors) = adj.get(&node) {
                    for &nb in neighbors {
                        if !reachable.contains(&nb) {
                            reachable.insert(nb);
                            next_frontier.insert(nb);
                        }
                    }
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        // Build the sub-graph
        let mut sub = KnowledgeGraph::with_embedding_dim(self.embedding_dim);
        for &id in &reachable {
            if let Some(name) = self.id_to_name.get(id) {
                let default_type = "Unknown".to_string();
                let type_label = self
                    .entity_types
                    .get(&id)
                    .and_then(|v| v.first())
                    .unwrap_or(&default_type);
                sub.add_entity(name, type_label);
                // Copy additional types
                if let Some(types) = self.entity_types.get(&id) {
                    for extra in types.iter().skip(1) {
                        sub.add_entity(name, extra);
                    }
                }
                // Copy properties
                if let Some(props) = self.properties.get(&id) {
                    for (k, v) in props {
                        let _ = sub.set_property(name, k, v);
                    }
                }
            }
        }
        // Copy triples whose both endpoints are in the subgraph
        for t in &self.relations {
            if reachable.contains(&t.subject) && reachable.contains(&t.object) {
                if let (Some(sn), Some(on)) = (
                    self.id_to_name.get(t.subject),
                    self.id_to_name.get(t.object),
                ) {
                    sub.add_relation(sn, &t.predicate, on, t.confidence);
                }
            }
        }
        sub
    }

    // -----------------------------------------------------------------------
    // Entity embeddings (TransE-style)
    // -----------------------------------------------------------------------

    /// Compute or return a cached TransE-style embedding for `entity`.
    ///
    /// The embeddings are trained via a lightweight SGD-based TransE loop:
    /// `h + r ≈ t`, where `r` is a per-relation vector.
    ///
    /// Returns a zero vector when the entity is unknown.
    pub fn entity_embedding(&self, entity: &str) -> Vec<f64> {
        let Some(&id) = self.entities.get(entity) else {
            return vec![0.0; self.embedding_dim];
        };
        self.embeddings
            .get(&id)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.embedding_dim])
    }

    /// Train TransE embeddings for all entities and relations using SGD.
    ///
    /// This is a pure-Rust, dependency-free implementation sufficient for
    /// small knowledge graphs (< 100K triples).
    ///
    /// # Parameters
    /// - `epochs`   – number of training epochs (default-friendly: 200)
    /// - `lr`       – learning rate (default: 0.01)
    /// - `margin`   – margin for the ranking loss (default: 1.0)
    pub fn train_embeddings(&mut self, epochs: usize, lr: f64, margin: f64) {
        let n = self.id_to_name.len();
        if n == 0 || self.relations.is_empty() {
            return;
        }
        let dim = self.embedding_dim;

        // Collect unique predicates
        let predicates: Vec<String> = {
            let mut seen: HashSet<String> = HashSet::new();
            for t in &self.relations {
                seen.insert(t.predicate.clone());
            }
            seen.into_iter().collect()
        };
        let pred_index: HashMap<&str, usize> = predicates
            .iter()
            .enumerate()
            .map(|(i, p)| (p.as_str(), i))
            .collect();
        let p = predicates.len();

        // Initialise entity and relation vectors using a simple LCG PRNG
        // (avoids any rand dependency while still being deterministic)
        let mut rng_state: u64 = 0x853c49e6748fea9b;
        let mut lcg_next = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((*state >> 33) ^ *state) as f64;
            (bits / u64::MAX as f64) * 2.0 - 1.0
        };

        // entity_emb: [n × dim], rel_emb: [p × dim]
        let mut entity_emb: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| lcg_next(&mut rng_state) * 0.1).collect())
            .collect();
        let mut rel_emb: Vec<Vec<f64>> = (0..p)
            .map(|_| (0..dim).map(|_| lcg_next(&mut rng_state) * 0.1).collect())
            .collect();

        // Normalize entity embeddings to unit L2
        let l2_normalize = |v: &mut Vec<f64>| {
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in v.iter_mut() {
                *x /= norm;
            }
        };
        for e in entity_emb.iter_mut() {
            l2_normalize(e);
        }

        let triples_snap: Vec<Triple> = self.relations.clone();
        let n_triples = triples_snap.len();

        for _epoch in 0..epochs {
            // Simple sequential SGD over all positive triples with random corruption
            for (idx, triple) in triples_snap.iter().enumerate() {
                let Some(&r_idx) = pred_index.get(triple.predicate.as_str()) else {
                    continue;
                };
                let h = triple.subject;
                let t = triple.object;

                // Corrupt the head using a deterministic schedule
                let corrupt_head = idx % 2 == 0;
                let corrupt_id = if corrupt_head {
                    (idx * 1_000_003 + h + 7) % n
                } else {
                    (idx * 1_000_003 + t + 13) % n
                };
                // Ensure the corrupt triple differs from the positive one
                let (neg_h, neg_t) = if corrupt_head {
                    if corrupt_id == h {
                        ((corrupt_id + 1) % n, t)
                    } else {
                        (corrupt_id, t)
                    }
                } else {
                    if corrupt_id == t {
                        (h, (corrupt_id + 1) % n)
                    } else {
                        (h, corrupt_id)
                    }
                };

                // Compute L1 distances for positive and negative triples
                let pos_dist: f64 = (0..dim)
                    .map(|d| (entity_emb[h][d] + rel_emb[r_idx][d] - entity_emb[t][d]).abs())
                    .sum();
                let neg_dist: f64 = (0..dim)
                    .map(|d| (entity_emb[neg_h][d] + rel_emb[r_idx][d] - entity_emb[neg_t][d]).abs())
                    .sum();

                let loss = (margin + pos_dist - neg_dist).max(0.0);
                if loss == 0.0 {
                    continue;
                }

                // Gradient step: sign-based gradient for L1 loss
                for d in 0..dim {
                    let pos_sign = (entity_emb[h][d] + rel_emb[r_idx][d] - entity_emb[t][d])
                        .signum();
                    let neg_sign = (entity_emb[neg_h][d] + rel_emb[r_idx][d]
                        - entity_emb[neg_t][d])
                        .signum();

                    entity_emb[h][d] -= lr * pos_sign;
                    entity_emb[t][d] += lr * pos_sign;
                    rel_emb[r_idx][d] -= lr * pos_sign;

                    entity_emb[neg_h][d] += lr * neg_sign;
                    entity_emb[neg_t][d] -= lr * neg_sign;
                    // rel_emb not updated for negative (standard TransE)
                }

                // Re-normalise entity vectors involved in the update
                l2_normalize(&mut entity_emb[h]);
                l2_normalize(&mut entity_emb[t]);
                l2_normalize(&mut entity_emb[neg_h]);
                l2_normalize(&mut entity_emb[neg_t]);
            }
        }

        // Store back into the graph's embedding cache
        for (id, emb) in entity_emb.into_iter().enumerate() {
            self.embeddings.insert(id, emb);
        }
    }

    /// Return a reference to all triples.
    pub fn all_triples(&self) -> &[Triple] {
        &self.relations
    }

    /// Merge another knowledge graph into this one.
    pub fn merge(&mut self, other: &KnowledgeGraph) {
        for name in other.entities() {
            let types = other.entity_types(name);
            let first_type = types.first().copied().unwrap_or("Unknown");
            let id = self.add_entity(name, first_type);
            for &extra in types.iter().skip(1) {
                let types_ref = self.entity_types.entry(id).or_default();
                if !types_ref.contains(&extra.to_string()) {
                    types_ref.push(extra.to_string());
                }
            }
        }
        for t in &other.relations {
            if let (Some(sn), Some(on)) = (
                other.id_to_name.get(t.subject),
                other.id_to_name.get(t.object),
            ) {
                self.add_relation(sn, &t.predicate, on, t.confidence);
            }
        }
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn build_sample_graph() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("Alice", "Person");
        kg.add_entity("Bob", "Person");
        kg.add_entity("Acme", "Organization");
        kg.add_entity("London", "Location");
        kg.add_relation("Alice", "works_at", "Acme", 0.9);
        kg.add_relation("Bob", "works_at", "Acme", 0.85);
        kg.add_relation("Acme", "located_in", "London", 0.99);
        kg
    }

    #[test]
    fn test_add_and_query_entities() {
        let mut kg = KnowledgeGraph::new();
        let id1 = kg.add_entity("Alice", "Person");
        let id2 = kg.add_entity("Alice", "Employee"); // second type
        assert_eq!(id1, id2, "same entity should return same id");
        let types = kg.entity_types("Alice");
        assert!(types.contains(&"Person"));
        assert!(types.contains(&"Employee"));
    }

    #[test]
    fn test_query_relations() {
        let kg = build_sample_graph();
        let rels = kg.query_relations("Alice");
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].predicate, "works_at");
    }

    #[test]
    fn test_shortest_path() {
        let kg = build_sample_graph();
        let path = kg.shortest_path("Alice", "London").expect("path should exist");
        // Alice → Acme → London
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "Alice");
        assert_eq!(path[2], "London");
    }

    #[test]
    fn test_shortest_path_same_node() {
        let kg = build_sample_graph();
        let path = kg.shortest_path("Alice", "Alice").expect("trivial path");
        assert_eq!(path, vec!["Alice"]);
    }

    #[test]
    fn test_subgraph() {
        let kg = build_sample_graph();
        let sub = kg.subgraph("Acme", 1);
        // Acme + its 1-hop neighbours
        let names: Vec<&str> = sub.entities().collect();
        assert!(names.contains(&"Acme"));
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
        assert!(names.contains(&"London"));
    }

    #[test]
    fn test_entity_embedding_train() {
        let mut kg = build_sample_graph();
        kg.train_embeddings(20, 0.01, 1.0);
        let emb = kg.entity_embedding("Alice");
        assert_eq!(emb.len(), 64);
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.05, "embedding should be unit-norm");
    }

    #[test]
    fn test_properties() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("Alice", "Person");
        kg.set_property("Alice", "age", "30").expect("set_property failed");
        assert_eq!(kg.get_property("Alice", "age"), Some("30"));
        assert_eq!(kg.get_property("Alice", "missing"), None);
    }

    #[test]
    fn test_merge() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_entity("A", "Person");
        kg1.add_entity("B", "Organization");
        kg1.add_relation("A", "member_of", "B", 0.8);

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_entity("B", "Organization");
        kg2.add_entity("C", "Location");
        kg2.add_relation("B", "located_in", "C", 0.9);

        kg1.merge(&kg2);
        assert_eq!(kg1.num_entities(), 3);
        assert_eq!(kg1.num_triples(), 2);
    }
}
