//! Push-pull dataflow network.
//!
//! A dataflow network is a directed acyclic graph (DAG) of processing nodes.
//! Data flows from [`Source`] nodes, through transformation nodes ([`Map`],
//! [`Filter`], [`Zip`], [`Buffer`]), and is consumed by [`Sink`] nodes.
//!
//! ## Execution model
//!
//! The network is **lazy on the pull side** and **eager on the push side**:
//!
//! - Each node implements the [`DataflowNode<T>`] trait that exposes both
//!   `push` (send a value downstream) and `pull` (request the next value from
//!   upstream).
//! - [`DataflowGraph`] manages a set of named nodes and their connections.
//!   Calling [`DataflowGraph::run`] drains all `Source` nodes, pushing every
//!   available value through the graph in topological order.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::reactive::dataflow::{DataflowGraph, Source, Map, Filter, Sink};
//!
//! let mut graph = DataflowGraph::new();
//! let src = Source::from_iter(0..10i32);
//! let map = Map::new(|x: i32| x * 2);
//! let filter = Filter::new(|x: &i32| *x > 8);
//! let sink: Sink<i32> = Sink::new();
//!
//! let src_id = graph.add_source(src);
//! let map_id = graph.add_map(map);
//! let flt_id = graph.add_filter(filter);
//! let snk_id = graph.add_sink(sink);
//!
//! graph.connect(src_id, map_id);
//! graph.connect(map_id, flt_id);
//! graph.connect(flt_id, snk_id);
//!
//! graph.run();
//!
//! let results = graph.collect_sink(snk_id);
//! assert_eq!(results, vec![10, 12, 14, 16, 18]);
//! ```

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Opaque node identifier within a [`DataflowGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

// ---------------------------------------------------------------------------
// DataflowNode trait (type-erased)
// ---------------------------------------------------------------------------

/// Trait for nodes in the dataflow graph.
///
/// Both `push` and `pull` are provided so a node can act as either a consumer
/// (pull from upstream, push downstream) or a pure transformer.
pub trait DataflowNode<T>: Send + Sync {
    /// Push a value into this node.  The node may buffer or forward it.
    fn push(&self, value: T);

    /// Pull the next available value from this node's output queue.
    fn pull(&self) -> Option<T>;
}

// ---------------------------------------------------------------------------
// Source<T>
// ---------------------------------------------------------------------------

struct SourceInner<T> {
    buffer: std::collections::VecDeque<T>,
}

/// A source node that produces values from an iterator or via manual push.
pub struct Source<T> {
    inner: Arc<Mutex<SourceInner<T>>>,
}

impl<T: Clone + Send + Sync + 'static> Source<T> {
    /// Create a source pre-loaded with values from an iterator.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter(iter: impl Iterator<Item = T>) -> Self {
        let buffer: std::collections::VecDeque<T> = iter.collect();
        Source {
            inner: Arc::new(Mutex::new(SourceInner { buffer })),
        }
    }

    /// Create an empty source (values added via `push_value`).
    pub fn empty() -> Self {
        Source {
            inner: Arc::new(Mutex::new(SourceInner {
                buffer: std::collections::VecDeque::new(),
            })),
        }
    }

    /// Manually push a value into the source buffer.
    pub fn push_value(&self, value: T) {
        if let Ok(mut g) = self.inner.lock() {
            g.buffer.push_back(value);
        }
    }

    /// Number of buffered values.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.buffer.len()).unwrap_or(0)
    }

    /// `true` if the source buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone + Send + Sync + 'static> DataflowNode<T> for Source<T> {
    fn push(&self, value: T) {
        if let Ok(mut g) = self.inner.lock() {
            g.buffer.push_back(value);
        }
    }

    fn pull(&self) -> Option<T> {
        self.inner.lock().ok()?.buffer.pop_front()
    }
}

// ---------------------------------------------------------------------------
// Sink<T>
// ---------------------------------------------------------------------------

struct SinkInner<T> {
    collected: Vec<T>,
    callback: Option<Box<dyn Fn(T) + Send + Sync + 'static>>,
}

/// A sink node that consumes values and optionally applies a callback.
pub struct Sink<T> {
    inner: Arc<Mutex<SinkInner<T>>>,
}

impl<T: Clone + Send + Sync + 'static> Sink<T> {
    /// Create a collecting sink (values are stored in an internal `Vec`).
    pub fn new() -> Self {
        Sink {
            inner: Arc::new(Mutex::new(SinkInner {
                collected: Vec::new(),
                callback: None,
            })),
        }
    }

    /// Create a sink that applies `f` to every received value.
    pub fn for_each(f: impl Fn(T) + Send + Sync + 'static) -> Self {
        Sink {
            inner: Arc::new(Mutex::new(SinkInner {
                collected: Vec::new(),
                callback: Some(Box::new(f)),
            })),
        }
    }

    /// Return the collected values, leaving the internal buffer empty.
    pub fn drain(&self) -> Vec<T> {
        self.inner
            .lock()
            .map(|mut g| std::mem::take(&mut g.collected))
            .unwrap_or_default()
    }

    /// Return a clone of collected values without draining.
    pub fn collect(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.inner
            .lock()
            .map(|g| g.collected.clone())
            .unwrap_or_default()
    }
}

impl<T: Clone + Send + Sync + 'static> DataflowNode<T> for Sink<T> {
    fn push(&self, value: T) {
        if let Ok(mut g) = self.inner.lock() {
            if let Some(cb) = &g.callback {
                cb(value);
            } else {
                g.collected.push(value);
            }
        }
    }

    fn pull(&self) -> Option<T> {
        // Sinks do not produce values.
        None
    }
}

// ---------------------------------------------------------------------------
// Map<T, U>
// ---------------------------------------------------------------------------

/// A transformation node that applies a function `T -> U`.
pub struct Map<T, U> {
    func: Arc<dyn Fn(T) -> U + Send + Sync + 'static>,
    output: Arc<Mutex<std::collections::VecDeque<U>>>,
}

impl<T: Send + Sync + 'static, U: Send + Sync + 'static> Map<T, U> {
    /// Create a `Map` node with the given transformation function.
    pub fn new(f: impl Fn(T) -> U + Send + Sync + 'static) -> Self {
        Map {
            func: Arc::new(f),
            output: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        }
    }
}

impl<T: Send + Sync + 'static, U: Send + Sync + 'static> DataflowNode<T> for Map<T, U> {
    fn push(&self, value: T) {
        let out = (self.func)(value);
        if let Ok(mut q) = self.output.lock() {
            q.push_back(out);
        }
    }

    fn pull(&self) -> Option<T> {
        // Map produces U, not T; this method is not meaningful here.
        // It is only called on Map<T,T> or via the graph which uses
        // the type-erased output queue directly.
        None
    }
}

impl<T: Send + Sync + 'static, U: Send + Sync + 'static> Map<T, U> {
    /// Pull from the output queue (produces `U`).
    pub fn pull_out(&self) -> Option<U> {
        self.output.lock().ok()?.pop_front()
    }
}

// ---------------------------------------------------------------------------
// Filter<T>
// ---------------------------------------------------------------------------

/// A filter node that forwards values satisfying a predicate.
pub struct Filter<T> {
    pred: Arc<dyn Fn(&T) -> bool + Send + Sync + 'static>,
    output: Arc<Mutex<std::collections::VecDeque<T>>>,
}

impl<T: Send + Sync + 'static> Filter<T> {
    /// Create a `Filter` node with the given predicate.
    pub fn new(pred: impl Fn(&T) -> bool + Send + Sync + 'static) -> Self {
        Filter {
            pred: Arc::new(pred),
            output: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        }
    }
}

impl<T: Send + Sync + 'static> DataflowNode<T> for Filter<T> {
    fn push(&self, value: T) {
        if (self.pred)(&value) {
            if let Ok(mut q) = self.output.lock() {
                q.push_back(value);
            }
        }
    }

    fn pull(&self) -> Option<T> {
        self.output.lock().ok()?.pop_front()
    }
}

// ---------------------------------------------------------------------------
// Zip<T, U>
// ---------------------------------------------------------------------------

/// A zip node that combines values from two input queues into pairs.
///
/// A pair `(T, U)` is emitted only when both queues have at least one element.
pub struct Zip<T, U> {
    left: Arc<Mutex<std::collections::VecDeque<T>>>,
    right: Arc<Mutex<std::collections::VecDeque<U>>>,
    output: Arc<Mutex<std::collections::VecDeque<(T, U)>>>,
}

impl<T: Send + Sync + 'static, U: Send + Sync + 'static> Zip<T, U> {
    /// Create a new `Zip` node.
    pub fn new() -> Self {
        Zip {
            left: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            right: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            output: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        }
    }

    /// Push a value for the **left** stream.
    pub fn push_left(&self, value: T) {
        if let Ok(mut l) = self.left.lock() {
            l.push_back(value);
        }
        self.try_pair();
    }

    /// Push a value for the **right** stream.
    pub fn push_right(&self, value: U) {
        if let Ok(mut r) = self.right.lock() {
            r.push_back(value);
        }
        self.try_pair();
    }

    fn try_pair(&self) {
        loop {
            let pair = {
                let mut l = match self.left.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                let mut r = match self.right.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                match (l.pop_front(), r.pop_front()) {
                    (Some(lv), Some(rv)) => (lv, rv),
                    (Some(lv), None) => {
                        l.push_front(lv);
                        break;
                    }
                    (None, Some(rv)) => {
                        r.push_front(rv);
                        break;
                    }
                    (None, None) => break,
                }
            };
            if let Ok(mut out) = self.output.lock() {
                out.push_back(pair);
            }
        }
    }

    /// Pull the next `(T, U)` pair.
    pub fn pull_pair(&self) -> Option<(T, U)> {
        self.output.lock().ok()?.pop_front()
    }
}

// ---------------------------------------------------------------------------
// Buffer<T>
// ---------------------------------------------------------------------------

/// A buffering node that accumulates `batch_size` items before releasing them
/// as a batch (`Vec<T>`).
pub struct Buffer<T> {
    batch_size: usize,
    input: Arc<Mutex<std::collections::VecDeque<T>>>,
    output: Arc<Mutex<std::collections::VecDeque<Vec<T>>>>,
}

impl<T: Send + Sync + 'static> Buffer<T> {
    /// Create a `Buffer` that emits batches of `batch_size` items.
    pub fn new(batch_size: usize) -> Self {
        Buffer {
            batch_size: batch_size.max(1),
            input: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            output: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        }
    }

    fn flush_if_ready(&self) {
        loop {
            let batch: Option<Vec<T>> = {
                let mut inp = match self.input.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                if inp.len() >= self.batch_size {
                    Some(inp.drain(..self.batch_size).collect())
                } else {
                    None
                }
            };
            match batch {
                Some(b) => {
                    if let Ok(mut out) = self.output.lock() {
                        out.push_back(b);
                    }
                }
                None => break,
            }
        }
    }

    /// Pull the next complete batch.
    pub fn pull_batch(&self) -> Option<Vec<T>> {
        self.output.lock().ok()?.pop_front()
    }

    /// Number of complete batches available.
    pub fn batch_count(&self) -> usize {
        self.output.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl<T: Send + Sync + 'static> DataflowNode<T> for Buffer<T> {
    fn push(&self, value: T) {
        if let Ok(mut inp) = self.input.lock() {
            inp.push_back(value);
        }
        self.flush_if_ready();
    }

    fn pull(&self) -> Option<T> {
        // Buffer exposes batches, not individual items.
        None
    }
}

// ---------------------------------------------------------------------------
// DataflowGraph
// ---------------------------------------------------------------------------

/// Type-erased node wrapper stored inside the graph.
enum AnyNode {
    SourceI32(Arc<Source<i32>>),
    SinkI32(Arc<Sink<i32>>),
    MapI32(Arc<Map<i32, i32>>),
    FilterI32(Arc<Filter<i32>>),
    BufferI32(Arc<Buffer<i32>>),
    SourceF64(Arc<Source<f64>>),
    SinkF64(Arc<Sink<f64>>),
}

/// A typed node descriptor for building the graph with [`DataflowGraph`].
///
/// The graph stores nodes as typed variants internally; connections between
/// nodes of compatible types are resolved at `run()` time.
#[allow(missing_debug_implementations)]
pub struct DataflowGraph {
    nodes: Vec<AnyNode>,
    edges: Vec<(NodeId, NodeId)>,
}

impl DataflowGraph {
    /// Create a new, empty graph.
    pub fn new() -> Self {
        DataflowGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    // --- i32 nodes ---------------------------------------------------------

    /// Add an `i32` source node and return its [`NodeId`].
    pub fn add_source(&mut self, src: Source<i32>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::SourceI32(Arc::new(src)));
        id
    }

    /// Add an `i32 → i32` map node and return its [`NodeId`].
    pub fn add_map(&mut self, map: Map<i32, i32>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::MapI32(Arc::new(map)));
        id
    }

    /// Add an `i32` filter node and return its [`NodeId`].
    pub fn add_filter(&mut self, filter: Filter<i32>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::FilterI32(Arc::new(filter)));
        id
    }

    /// Add an `i32` sink node and return its [`NodeId`].
    pub fn add_sink(&mut self, sink: Sink<i32>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::SinkI32(Arc::new(sink)));
        id
    }

    /// Add an `i32` buffer node and return its [`NodeId`].
    pub fn add_buffer(&mut self, buf: Buffer<i32>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::BufferI32(Arc::new(buf)));
        id
    }

    // --- f64 nodes ---------------------------------------------------------

    /// Add an `f64` source node and return its [`NodeId`].
    pub fn add_source_f64(&mut self, src: Source<f64>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::SourceF64(Arc::new(src)));
        id
    }

    /// Add an `f64` sink node and return its [`NodeId`].
    pub fn add_sink_f64(&mut self, sink: Sink<f64>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(AnyNode::SinkF64(Arc::new(sink)));
        id
    }

    // --- edges -------------------------------------------------------------

    /// Connect the output of `src` to the input of `dst`.
    pub fn connect(&mut self, src: NodeId, dst: NodeId) {
        self.edges.push((src, dst));
    }

    // --- execution ---------------------------------------------------------

    /// Drive all source nodes and push values through the graph until all
    /// sources are empty.
    pub fn run(&self) {
        // Build an adjacency list: for each node, which nodes receive its output?
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for &(NodeId(src), NodeId(dst)) in &self.edges {
            if src < adjacency.len() {
                adjacency[src].push(dst);
            }
        }

        // Pull from every source and propagate.
        let mut changed = true;
        while changed {
            changed = false;
            for (src_idx, node) in self.nodes.iter().enumerate() {
                match node {
                    AnyNode::SourceI32(src) => {
                        while let Some(v) = src.pull() {
                            changed = true;
                            self.propagate_i32(v, &adjacency[src_idx]);
                        }
                    }
                    AnyNode::MapI32(map) => {
                        while let Some(v) = map.pull_out() {
                            changed = true;
                            self.propagate_i32(v, &adjacency[src_idx]);
                        }
                    }
                    AnyNode::FilterI32(flt) => {
                        while let Some(v) = flt.pull() {
                            changed = true;
                            self.propagate_i32(v, &adjacency[src_idx]);
                        }
                    }
                    AnyNode::SourceF64(src) => {
                        while let Some(v) = src.pull() {
                            changed = true;
                            self.propagate_f64(v, &adjacency[src_idx]);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn propagate_i32(&self, value: i32, dst_indices: &[usize]) {
        for &dst in dst_indices {
            match self.nodes.get(dst) {
                Some(AnyNode::MapI32(map)) => map.push(value),
                Some(AnyNode::FilterI32(flt)) => flt.push(value),
                Some(AnyNode::SinkI32(sink)) => sink.push(value),
                Some(AnyNode::BufferI32(buf)) => buf.push(value),
                _ => {}
            }
        }
    }

    fn propagate_f64(&self, value: f64, dst_indices: &[usize]) {
        for &dst in dst_indices {
            if let Some(AnyNode::SinkF64(sink)) = self.nodes.get(dst) {
                sink.push(value)
            }
        }
    }

    /// Collect and drain all values accumulated in the `i32` sink at `id`.
    pub fn collect_sink(&self, id: NodeId) -> Vec<i32> {
        match self.nodes.get(id.0) {
            Some(AnyNode::SinkI32(sink)) => sink.drain(),
            _ => Vec::new(),
        }
    }

    /// Collect and drain all values accumulated in the `f64` sink at `id`.
    pub fn collect_sink_f64(&self, id: NodeId) -> Vec<f64> {
        match self.nodes.get(id.0) {
            Some(AnyNode::SinkF64(sink)) => sink.drain(),
            _ => Vec::new(),
        }
    }

    /// Return available batches from a `Buffer` node at `id`.
    pub fn collect_buffer(&self, id: NodeId) -> Vec<Vec<i32>> {
        match self.nodes.get(id.0) {
            Some(AnyNode::BufferI32(buf)) => {
                let mut batches = Vec::new();
                while let Some(b) = buf.pull_batch() {
                    batches.push(b);
                }
                batches
            }
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataflow_map() {
        let mut graph = DataflowGraph::new();
        let src = Source::from_iter(0..5i32);
        let map = Map::new(|x: i32| x * 3);
        let sink: Sink<i32> = Sink::new();

        let src_id = graph.add_source(src);
        let map_id = graph.add_map(map);
        let snk_id = graph.add_sink(sink);

        graph.connect(src_id, map_id);
        graph.connect(map_id, snk_id);
        graph.run();

        let res = graph.collect_sink(snk_id);
        assert_eq!(res, vec![0, 3, 6, 9, 12]);
    }

    #[test]
    fn test_dataflow_filter() {
        let mut graph = DataflowGraph::new();
        let src = Source::from_iter(0..10i32);
        let flt = Filter::new(|x: &i32| x % 2 == 0);
        let sink: Sink<i32> = Sink::new();

        let src_id = graph.add_source(src);
        let flt_id = graph.add_filter(flt);
        let snk_id = graph.add_sink(sink);

        graph.connect(src_id, flt_id);
        graph.connect(flt_id, snk_id);
        graph.run();

        let res = graph.collect_sink(snk_id);
        assert_eq!(res, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_dataflow_source_sink() {
        let mut graph = DataflowGraph::new();
        let src = Source::from_iter(1..=5i32);
        let sink: Sink<i32> = Sink::new();

        let src_id = graph.add_source(src);
        let snk_id = graph.add_sink(sink);

        graph.connect(src_id, snk_id);
        graph.run();

        let res = graph.collect_sink(snk_id);
        assert_eq!(res, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_dataflow_buffer() {
        let mut graph = DataflowGraph::new();
        let src = Source::from_iter(0..9i32);
        let buf = Buffer::new(3);
        let src_id = graph.add_source(src);
        let snk_buf_id = graph.add_buffer(buf);

        graph.connect(src_id, snk_buf_id);
        graph.run();

        let batches = graph.collect_buffer(snk_buf_id);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
    }

    #[test]
    fn test_dataflow_zip() {
        let zip: Zip<i32, i32> = Zip::new();
        zip.push_left(1);
        zip.push_left(2);
        zip.push_right(10);
        zip.push_right(20);
        zip.push_left(3);
        zip.push_right(30);

        let mut pairs = Vec::new();
        while let Some(p) = zip.pull_pair() {
            pairs.push(p);
        }
        assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn test_source_manual_push() {
        let src: Source<i32> = Source::empty();
        src.push_value(5);
        src.push_value(6);
        assert_eq!(src.pull(), Some(5));
        assert_eq!(src.pull(), Some(6));
        assert_eq!(src.pull(), None);
    }

    #[test]
    fn test_sink_collect() {
        let sink: Sink<i32> = Sink::new();
        sink.push(1);
        sink.push(2);
        sink.push(3);
        assert_eq!(sink.collect(), vec![1, 2, 3]);
    }

    #[test]
    fn test_dataflow_map_filter_pipeline() {
        let mut graph = DataflowGraph::new();
        let src = Source::from_iter(0..10i32);
        let map = Map::new(|x: i32| x * 2);
        let filter = Filter::new(|x: &i32| *x > 8);
        let sink: Sink<i32> = Sink::new();

        let src_id = graph.add_source(src);
        let map_id = graph.add_map(map);
        let flt_id = graph.add_filter(filter);
        let snk_id = graph.add_sink(sink);

        graph.connect(src_id, map_id);
        graph.connect(map_id, flt_id);
        graph.connect(flt_id, snk_id);
        graph.run();

        let res = graph.collect_sink(snk_id);
        assert_eq!(res, vec![10, 12, 14, 16, 18]);
    }
}
