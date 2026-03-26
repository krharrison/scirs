// API Stability Tests for scirs2-core
//
// These tests verify that the core public API surface has not been accidentally
// broken. If this file fails to compile, a previously-stable public item has
// been removed or its signature changed in a backward-incompatible way.
//
// Guidelines:
// - Each test function covers one logical group of public items.
// - Only imports/type-checks that compile are required — no logic needed.
// - No `unwrap()` — use `expect("…")` for any fallible calls.
// - New tests should be added whenever a stabilised public item is introduced.

// ---------------------------------------------------------------------------
// Error module
// ---------------------------------------------------------------------------

/// Verify that the key error types and result aliases are accessible.
#[test]
fn test_error_types_accessible() {
    use scirs2_core::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

    // CoreError can be constructed in several canonical variants.
    let _computation_err = CoreError::ComputationError(ErrorContext::new("test"));
    let _memory_err = CoreError::MemoryError(ErrorContext::new("test"));
    // InvalidInput also takes ErrorContext, not String
    let _validation_err = CoreError::InvalidInput(ErrorContext::new("bad input"));

    // ErrorContext builder is chainable.
    let ctx = ErrorContext::new("test context").with_location(ErrorLocation::new(file!(), line!()));
    let _ = ctx;

    // CoreResult is usable as a return type.
    fn _returns_core_result() -> CoreResult<i32> {
        Ok(42)
    }
    let v = _returns_core_result().expect("should succeed");
    assert_eq!(v, 42);
}

/// Verify that recovery helpers are accessible.
#[test]
fn test_error_recovery_accessible() {
    use scirs2_core::error::{
        CoreError, ErrorContext, ErrorSeverity, RecoverableError, RecoveryHint, RecoveryStrategy,
    };

    let err = CoreError::InvalidInput(ErrorContext::new("test"));
    let recoverable = RecoverableError::error(err);
    let _ = recoverable.recovery_report();

    // Smoke-test that these enum/struct variants exist and are nameable.
    let _severity: ErrorSeverity = ErrorSeverity::Critical;
    // RecoveryHint is a struct, not an enum
    let _hint = RecoveryHint::new("retry", "try again", 0.8);
    // RecoveryStrategy variants
    let _strategy = RecoveryStrategy::FailFast;
    let _strategy2 = RecoveryStrategy::Fallback;
}

// ---------------------------------------------------------------------------
// Validation module
// ---------------------------------------------------------------------------

/// Verify that validation helpers are accessible.
#[test]
fn test_validation_accessible() {
    use scirs2_core::validation::{check_non_negative, check_positive};

    let ok_result = check_positive(5usize, "count");
    assert!(
        ok_result.is_ok(),
        "check_positive should succeed for positive value"
    );

    let err_result = check_positive(0usize, "count");
    assert!(err_result.is_err(), "check_positive should fail for zero");

    let ok_non_neg = check_non_negative(0.0f64, "x");
    assert!(
        ok_non_neg.is_ok(),
        "check_non_negative should succeed for zero"
    );
}

// ---------------------------------------------------------------------------
// API versioning module
// ---------------------------------------------------------------------------

/// Verify that the Version type and its constants/methods are accessible.
#[test]
fn test_apiversioning_accessible() {
    use scirs2_core::apiversioning::Version;

    let v = Version::new(0, 4, 0);
    assert_eq!(v.major, 0);
    assert_eq!(v.minor, 4);
    assert_eq!(v.patch, 0);

    let parsed = Version::parse("0.4.0").expect("valid semver string should parse");
    assert_eq!(parsed.major, 0);

    let current = Version::CURRENT;
    let _ = current.to_string();

    // Compatibility check API exists.
    let older = Version::new(0, 3, 0);
    let newer = Version::new(0, 5, 0);
    assert!(!older.is_compatible_with(&newer));
}

// ---------------------------------------------------------------------------
// Constants module
// ---------------------------------------------------------------------------

/// Verify that mathematical and physical constants are accessible.
#[test]
fn test_constants_accessible() {
    use scirs2_core::constants::{math, physical};

    let pi = math::PI;
    assert!((pi - std::f64::consts::PI).abs() < 1e-10);

    let e = math::E;
    assert!((e - std::f64::consts::E).abs() < 1e-10);

    let c = physical::SPEED_OF_LIGHT;
    assert!(c > 0.0, "speed of light must be positive");

    let h = physical::PLANCK;
    assert!(h > 0.0, "Planck constant must be positive");
}

// ---------------------------------------------------------------------------
// ndarray re-export
// ---------------------------------------------------------------------------

/// Verify that the core ndarray re-export is accessible.
#[test]
fn test_ndarray_reexport_accessible() {
    use scirs2_core::ndarray::Array2;

    let arr: Array2<f64> = Array2::zeros((3, 3));
    assert_eq!(arr.shape(), &[3, 3]);
}

// ---------------------------------------------------------------------------
// SIMD operations module
// ---------------------------------------------------------------------------

/// Verify that PlatformCapabilities is accessible (always compiled).
#[test]
fn test_simd_ops_accessible() {
    use scirs2_core::simd_ops::PlatformCapabilities;

    let caps = PlatformCapabilities::detect();
    // Verify the struct is constructible and fields are accessible.
    let _has_simd: bool = caps.simd_available;
    let _has_gpu: bool = caps.gpu_available;
}

// ---------------------------------------------------------------------------
// Stability module (StabilityLevel enum)
// ---------------------------------------------------------------------------

/// Verify that the stability module types are accessible.
#[test]
fn test_stability_module_accessible() {
    use scirs2_core::stability::StabilityLevel;

    // All four canonical stability levels must remain nameable.
    let _stable = StabilityLevel::Stable;
    let _evolving = StabilityLevel::Evolving;
    let _experimental = StabilityLevel::Experimental;
    let _deprecated = StabilityLevel::Deprecated;
}

// ---------------------------------------------------------------------------
// Concurrent module
// ---------------------------------------------------------------------------

/// Verify that ConcurrentHashMap and BoundedQueue are accessible.
#[test]
fn test_concurrent_accessible() {
    use scirs2_core::concurrent::{BoundedQueue, ConcurrentHashMap};

    let map: ConcurrentHashMap<String, u64> = ConcurrentHashMap::new();
    map.insert("hello".into(), 99);
    let v = map.get(&"hello".to_string());
    assert_eq!(v, Some(99));

    let queue: BoundedQueue<i32> = BoundedQueue::new(16);
    queue
        .push(7)
        .expect("queue push should succeed on empty queue");
    let item = queue.pop();
    assert_eq!(item, Some(7));
}

// ---------------------------------------------------------------------------
// Task graph module
// ---------------------------------------------------------------------------

/// Verify that the TaskGraph type is accessible.
#[test]
fn test_task_graph_accessible() {
    use scirs2_core::task_graph::TaskGraph;

    // Constructing a TaskGraph with a unit payload type must compile.
    let _graph: TaskGraph<()> = TaskGraph::new();
}

// ---------------------------------------------------------------------------
// Collections module
// ---------------------------------------------------------------------------

/// Verify that the FlatMap and RrbVec types are accessible.
#[test]
fn test_collections_accessible() {
    use scirs2_core::collections::flat_map::FlatMap;
    use scirs2_core::collections::RrbVec;

    let mut map: FlatMap<String, u32> = FlatMap::new();
    map.insert("key".to_string(), 42);
    let val = map
        .get(&"key".to_string())
        .copied()
        .expect("inserted key should be present");
    assert_eq!(val, 42);

    // RrbVec (persistent radix-balanced tree vector backed by u64 words) must exist.
    let v: RrbVec = RrbVec::new();
    let v2 = v.push_back(10u64);
    assert_eq!(v2.len(), 1);
}

// ---------------------------------------------------------------------------
// Reactive module
// ---------------------------------------------------------------------------

/// Verify that the core reactive stream types are accessible.
#[test]
fn test_reactive_accessible() {
    use scirs2_core::reactive::{InfiniteStream, Stream};

    let s = InfiniteStream::from_iter(0i32..10);
    // Use explicit iterator filter+take to avoid method ambiguity with Stream trait.
    let evens: Vec<i32> = s.into_inner().filter(|x| x % 2 == 0).take(3).collect();
    assert_eq!(evens, vec![0, 2, 4]);
}

/// Verify that the reactive Signal / Observable types are accessible.
#[test]
fn test_reactive_signal_accessible() {
    use scirs2_core::reactive::signal::Observable;

    let obs: Observable<i32> = Observable::new(0);
    assert_eq!(obs.get(), 0);
    obs.set(42);
    assert_eq!(obs.get(), 42);
}

// ---------------------------------------------------------------------------
// Arithmetic (double-double) module
// ---------------------------------------------------------------------------

/// Verify that the DoubleDouble type and helper functions are accessible.
#[test]
fn test_arithmetic_accessible() {
    use scirs2_core::arithmetic::{sum_dd, DoubleDouble};

    let dd = DoubleDouble::from(1.0f64);
    let result = dd + DoubleDouble::from(2.0f64);
    let as_f64: f64 = result.to_f64();
    assert!((as_f64 - 3.0).abs() < 1e-14);

    // Compensated summation helper.
    let values = [1.0_f64, 1e100, 1.0, -1e100];
    let sum = sum_dd(&values);
    assert!((sum.to_f64() - 2.0).abs() < 1e-10);
}
