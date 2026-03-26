// This should fail because CoreError is #[non_exhaustive] and must not be matched exhaustively
// without a wildcard arm.
use scirs2_core::error::CoreError;

fn check_non_exhaustive(e: CoreError) {
    match e {
        CoreError::ComputationError(_) => {}
        CoreError::DomainError(_) => {}
        // Intentionally missing a wildcard arm — must fail with E0004
    }
}

fn main() {}
