//! ErrorKind enum for programmatic error matching
//!
//! Provides a lightweight, copyable enum that mirrors `CoreError` variants
//! without carrying payload, enabling efficient `match` / `if`-based dispatch.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_core::error::{CoreError, ErrorContext};
//! use scirs2_core::error::error_kind::ErrorKind;
//!
//! let err = CoreError::DimensionError(ErrorContext::new("shape mismatch"));
//! assert_eq!(err.kind(), ErrorKind::Dimension);
//! assert!(err.kind().is_shape_related());
//! ```

use super::error::CoreError;

/// Lightweight classification of `CoreError` variants.
///
/// Every `CoreError` variant maps to exactly one `ErrorKind`.
/// This enum is `Copy` and suitable for use in `match` arms, hash maps, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Generic computation error
    Computation,
    /// Input outside valid mathematical domain
    Domain,
    /// Array protocol dispatch failure
    Dispatch,
    /// Algorithm did not converge
    Convergence,
    /// Array dimension mismatch
    Dimension,
    /// Array shape incompatibility
    Shape,
    /// Index out of bounds
    Index,
    /// Invalid value
    Value,
    /// Type mismatch
    Type,
    /// Feature not implemented
    NotImplemented,
    /// Partial implementation
    Implementation,
    /// Memory allocation failure
    Memory,
    /// Allocation-specific error
    Allocation,
    /// Configuration error
    Config,
    /// Invalid argument
    InvalidArgument,
    /// Invalid input
    InvalidInput,
    /// Permission error
    Permission,
    /// Validation failure
    Validation,
    /// Invalid object state
    InvalidState,
    /// JIT compilation error
    Jit,
    /// JSON error
    Json,
    /// I/O error
    Io,
    /// Scheduler error
    Scheduler,
    /// Timeout error
    Timeout,
    /// Compression error
    Compression,
    /// Invalid array shape
    InvalidShape,
    /// GPU/device error
    Device,
    /// Mutex/lock error
    Mutex,
    /// Thread error
    Thread,
    /// Stream error
    Stream,
    /// End-of-stream
    EndOfStream,
    /// Resource error
    Resource,
    /// Communication error
    Communication,
    /// Security error
    Security,
}

impl ErrorKind {
    /// Human-readable name of this error kind.
    pub const fn as_str(&self) -> &'static str {
        match self {
            ErrorKind::Computation => "computation",
            ErrorKind::Domain => "domain",
            ErrorKind::Dispatch => "dispatch",
            ErrorKind::Convergence => "convergence",
            ErrorKind::Dimension => "dimension",
            ErrorKind::Shape => "shape",
            ErrorKind::Index => "index",
            ErrorKind::Value => "value",
            ErrorKind::Type => "type",
            ErrorKind::NotImplemented => "not_implemented",
            ErrorKind::Implementation => "implementation",
            ErrorKind::Memory => "memory",
            ErrorKind::Allocation => "allocation",
            ErrorKind::Config => "config",
            ErrorKind::InvalidArgument => "invalid_argument",
            ErrorKind::InvalidInput => "invalid_input",
            ErrorKind::Permission => "permission",
            ErrorKind::Validation => "validation",
            ErrorKind::InvalidState => "invalid_state",
            ErrorKind::Jit => "jit",
            ErrorKind::Json => "json",
            ErrorKind::Io => "io",
            ErrorKind::Scheduler => "scheduler",
            ErrorKind::Timeout => "timeout",
            ErrorKind::Compression => "compression",
            ErrorKind::InvalidShape => "invalid_shape",
            ErrorKind::Device => "device",
            ErrorKind::Mutex => "mutex",
            ErrorKind::Thread => "thread",
            ErrorKind::Stream => "stream",
            ErrorKind::EndOfStream => "end_of_stream",
            ErrorKind::Resource => "resource",
            ErrorKind::Communication => "communication",
            ErrorKind::Security => "security",
        }
    }

    /// Whether this error kind is related to array shapes or dimensions.
    pub const fn is_shape_related(&self) -> bool {
        matches!(
            self,
            ErrorKind::Dimension | ErrorKind::Shape | ErrorKind::InvalidShape
        )
    }

    /// Whether this error kind is related to resource exhaustion.
    pub const fn is_resource_related(&self) -> bool {
        matches!(
            self,
            ErrorKind::Memory | ErrorKind::Allocation | ErrorKind::Resource | ErrorKind::Timeout
        )
    }

    /// Whether this error kind is related to I/O or communication.
    pub const fn is_io_related(&self) -> bool {
        matches!(
            self,
            ErrorKind::Io | ErrorKind::Stream | ErrorKind::EndOfStream | ErrorKind::Communication
        )
    }

    /// Whether this error kind is related to concurrency.
    pub const fn is_concurrency_related(&self) -> bool {
        matches!(
            self,
            ErrorKind::Mutex | ErrorKind::Thread | ErrorKind::Scheduler
        )
    }

    /// Whether this error kind indicates a programming error (e.g., invalid argument).
    pub const fn is_programming_error(&self) -> bool {
        matches!(
            self,
            ErrorKind::InvalidArgument
                | ErrorKind::InvalidInput
                | ErrorKind::InvalidState
                | ErrorKind::NotImplemented
                | ErrorKind::Implementation
        )
    }

    /// Whether this error is typically retryable.
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorKind::Timeout
                | ErrorKind::Resource
                | ErrorKind::Communication
                | ErrorKind::Mutex
                | ErrorKind::Device
        )
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// CoreError::kind() method
// ---------------------------------------------------------------------------

impl CoreError {
    /// Classify this error into an `ErrorKind` for programmatic matching.
    pub fn kind(&self) -> ErrorKind {
        match self {
            CoreError::ComputationError(_) => ErrorKind::Computation,
            CoreError::DomainError(_) => ErrorKind::Domain,
            CoreError::DispatchError(_) => ErrorKind::Dispatch,
            CoreError::ConvergenceError(_) => ErrorKind::Convergence,
            CoreError::DimensionError(_) => ErrorKind::Dimension,
            CoreError::ShapeError(_) => ErrorKind::Shape,
            CoreError::IndexError(_) => ErrorKind::Index,
            CoreError::ValueError(_) => ErrorKind::Value,
            CoreError::TypeError(_) => ErrorKind::Type,
            CoreError::NotImplementedError(_) => ErrorKind::NotImplemented,
            CoreError::ImplementationError(_) => ErrorKind::Implementation,
            CoreError::MemoryError(_) => ErrorKind::Memory,
            CoreError::AllocationError(_) => ErrorKind::Allocation,
            CoreError::ConfigError(_) => ErrorKind::Config,
            CoreError::InvalidArgument(_) => ErrorKind::InvalidArgument,
            CoreError::InvalidInput(_) => ErrorKind::InvalidInput,
            CoreError::PermissionError(_) => ErrorKind::Permission,
            CoreError::ValidationError(_) => ErrorKind::Validation,
            CoreError::InvalidState(_) => ErrorKind::InvalidState,
            CoreError::JITError(_) => ErrorKind::Jit,
            CoreError::JSONError(_) => ErrorKind::Json,
            CoreError::IoError(_) => ErrorKind::Io,
            CoreError::SchedulerError(_) => ErrorKind::Scheduler,
            CoreError::TimeoutError(_) => ErrorKind::Timeout,
            CoreError::CompressionError(_) => ErrorKind::Compression,
            CoreError::InvalidShape(_) => ErrorKind::InvalidShape,
            CoreError::DeviceError(_) => ErrorKind::Device,
            CoreError::MutexError(_) => ErrorKind::Mutex,
            CoreError::ThreadError(_) => ErrorKind::Thread,
            CoreError::StreamError(_) => ErrorKind::Stream,
            CoreError::EndOfStream(_) => ErrorKind::EndOfStream,
            CoreError::ResourceError(_) => ErrorKind::Resource,
            CoreError::CommunicationError(_) => ErrorKind::Communication,
            CoreError::SecurityError(_) => ErrorKind::Security,
        }
    }

    /// Extract the error message string from the inner `ErrorContext`.
    pub fn error_message(&self) -> &str {
        match self {
            CoreError::ComputationError(ctx)
            | CoreError::DomainError(ctx)
            | CoreError::DispatchError(ctx)
            | CoreError::ConvergenceError(ctx)
            | CoreError::DimensionError(ctx)
            | CoreError::ShapeError(ctx)
            | CoreError::IndexError(ctx)
            | CoreError::ValueError(ctx)
            | CoreError::TypeError(ctx)
            | CoreError::NotImplementedError(ctx)
            | CoreError::ImplementationError(ctx)
            | CoreError::MemoryError(ctx)
            | CoreError::AllocationError(ctx)
            | CoreError::ConfigError(ctx)
            | CoreError::InvalidArgument(ctx)
            | CoreError::InvalidInput(ctx)
            | CoreError::PermissionError(ctx)
            | CoreError::ValidationError(ctx)
            | CoreError::InvalidState(ctx)
            | CoreError::JITError(ctx)
            | CoreError::JSONError(ctx)
            | CoreError::IoError(ctx)
            | CoreError::SchedulerError(ctx)
            | CoreError::TimeoutError(ctx)
            | CoreError::CompressionError(ctx)
            | CoreError::InvalidShape(ctx)
            | CoreError::DeviceError(ctx)
            | CoreError::MutexError(ctx)
            | CoreError::ThreadError(ctx)
            | CoreError::StreamError(ctx)
            | CoreError::EndOfStream(ctx)
            | CoreError::ResourceError(ctx)
            | CoreError::CommunicationError(ctx)
            | CoreError::SecurityError(ctx) => &ctx.message,
        }
    }

    /// Get the inner `ErrorContext`.
    pub fn context(&self) -> &super::error::ErrorContext {
        match self {
            CoreError::ComputationError(ctx)
            | CoreError::DomainError(ctx)
            | CoreError::DispatchError(ctx)
            | CoreError::ConvergenceError(ctx)
            | CoreError::DimensionError(ctx)
            | CoreError::ShapeError(ctx)
            | CoreError::IndexError(ctx)
            | CoreError::ValueError(ctx)
            | CoreError::TypeError(ctx)
            | CoreError::NotImplementedError(ctx)
            | CoreError::ImplementationError(ctx)
            | CoreError::MemoryError(ctx)
            | CoreError::AllocationError(ctx)
            | CoreError::ConfigError(ctx)
            | CoreError::InvalidArgument(ctx)
            | CoreError::InvalidInput(ctx)
            | CoreError::PermissionError(ctx)
            | CoreError::ValidationError(ctx)
            | CoreError::InvalidState(ctx)
            | CoreError::JITError(ctx)
            | CoreError::JSONError(ctx)
            | CoreError::IoError(ctx)
            | CoreError::SchedulerError(ctx)
            | CoreError::TimeoutError(ctx)
            | CoreError::CompressionError(ctx)
            | CoreError::InvalidShape(ctx)
            | CoreError::DeviceError(ctx)
            | CoreError::MutexError(ctx)
            | CoreError::ThreadError(ctx)
            | CoreError::StreamError(ctx)
            | CoreError::EndOfStream(ctx)
            | CoreError::ResourceError(ctx)
            | CoreError::CommunicationError(ctx)
            | CoreError::SecurityError(ctx) => ctx,
        }
    }

    /// Return the chained cause, if any.
    pub fn cause(&self) -> Option<&CoreError> {
        self.context().cause.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Additional From implementations for cross-crate error conversion
// ---------------------------------------------------------------------------

impl From<std::num::ParseIntError> for CoreError {
    fn from(err: std::num::ParseIntError) -> Self {
        CoreError::ValueError(super::error::ErrorContext::new(format!(
            "Integer parse error: {err}"
        )))
    }
}

impl From<std::num::ParseFloatError> for CoreError {
    fn from(err: std::num::ParseFloatError) -> Self {
        CoreError::ValueError(super::error::ErrorContext::new(format!(
            "Float parse error: {err}"
        )))
    }
}

impl From<std::fmt::Error> for CoreError {
    fn from(err: std::fmt::Error) -> Self {
        CoreError::ComputationError(super::error::ErrorContext::new(format!(
            "Formatting error: {err}"
        )))
    }
}

impl From<std::str::Utf8Error> for CoreError {
    fn from(err: std::str::Utf8Error) -> Self {
        CoreError::ValueError(super::error::ErrorContext::new(format!(
            "UTF-8 decode error: {err}"
        )))
    }
}

impl From<std::string::FromUtf8Error> for CoreError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        CoreError::ValueError(super::error::ErrorContext::new(format!(
            "UTF-8 decode error: {err}"
        )))
    }
}

impl<T> From<std::sync::PoisonError<T>> for CoreError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        CoreError::MutexError(super::error::ErrorContext::new(format!(
            "Lock poisoned: {err}"
        )))
    }
}

impl From<std::time::SystemTimeError> for CoreError {
    fn from(err: std::time::SystemTimeError) -> Self {
        CoreError::ComputationError(super::error::ErrorContext::new(format!(
            "System time error: {err}"
        )))
    }
}

// ---------------------------------------------------------------------------
// ErrorContext backtrace support
// ---------------------------------------------------------------------------

impl super::error::ErrorContext {
    /// Capture a backtrace string and attach it to this context.
    ///
    /// Uses `std::backtrace::Backtrace` when available, otherwise
    /// produces a placeholder.
    pub fn with_backtrace(mut self) -> Self {
        // std::backtrace is stable since Rust 1.65
        let bt = std::backtrace::Backtrace::capture();
        let bt_string = format!("{bt}");
        if !bt_string.is_empty() && !bt_string.contains("disabled") {
            self.message = format!("{}\nBacktrace:\n{bt_string}", self.message);
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorContext;

    #[test]
    fn test_kind_mapping_computation() {
        let err = CoreError::ComputationError(ErrorContext::new("test"));
        assert_eq!(err.kind(), ErrorKind::Computation);
    }

    #[test]
    fn test_kind_mapping_dimension() {
        let err = CoreError::DimensionError(ErrorContext::new("bad shape"));
        assert_eq!(err.kind(), ErrorKind::Dimension);
        assert!(err.kind().is_shape_related());
    }

    #[test]
    fn test_kind_mapping_memory() {
        let err = CoreError::MemoryError(ErrorContext::new("oom"));
        assert_eq!(err.kind(), ErrorKind::Memory);
        assert!(err.kind().is_resource_related());
    }

    #[test]
    fn test_kind_mapping_io() {
        let err = CoreError::IoError(ErrorContext::new("file not found"));
        assert_eq!(err.kind(), ErrorKind::Io);
        assert!(err.kind().is_io_related());
    }

    #[test]
    fn test_kind_mapping_mutex() {
        let err = CoreError::MutexError(ErrorContext::new("poisoned"));
        assert_eq!(err.kind(), ErrorKind::Mutex);
        assert!(err.kind().is_concurrency_related());
        assert!(err.kind().is_retryable());
    }

    #[test]
    fn test_kind_as_str() {
        assert_eq!(ErrorKind::Computation.as_str(), "computation");
        assert_eq!(ErrorKind::Io.as_str(), "io");
        assert_eq!(ErrorKind::Security.as_str(), "security");
    }

    #[test]
    fn test_kind_display() {
        let kind = ErrorKind::Domain;
        assert_eq!(format!("{kind}"), "domain");
    }

    #[test]
    fn test_error_message_extraction() {
        let err = CoreError::ValueError(ErrorContext::new("bad value 42"));
        assert_eq!(err.error_message(), "bad value 42");
    }

    #[test]
    fn test_error_cause_chain() {
        let inner = CoreError::IoError(ErrorContext::new("disk full"));
        let outer =
            CoreError::ComputationError(ErrorContext::new("write failed").with_cause(inner));
        let cause = outer.cause();
        assert!(cause.is_some());
        assert_eq!(cause.map(|e| e.kind()), Some(ErrorKind::Io));
    }

    #[test]
    fn test_from_parse_int_error() {
        let result: Result<i32, _> = "not_a_number".parse();
        let err: CoreError = result.expect_err("should fail").into();
        assert_eq!(err.kind(), ErrorKind::Value);
    }

    #[test]
    fn test_from_parse_float_error() {
        let result: Result<f64, _> = "not_a_float".parse();
        let err: CoreError = result.expect_err("should fail").into();
        assert_eq!(err.kind(), ErrorKind::Value);
    }

    #[test]
    fn test_from_utf8_error() {
        let bytes = vec![0xFF, 0xFE];
        let result = String::from_utf8(bytes);
        let err: CoreError = result.expect_err("should fail").into();
        assert_eq!(err.kind(), ErrorKind::Value);
    }

    #[test]
    fn test_is_programming_error() {
        assert!(ErrorKind::InvalidArgument.is_programming_error());
        assert!(ErrorKind::NotImplemented.is_programming_error());
        assert!(!ErrorKind::Io.is_programming_error());
    }

    #[test]
    fn test_all_error_kinds_have_unique_as_str() {
        use std::collections::HashSet;
        let kinds = [
            ErrorKind::Computation,
            ErrorKind::Domain,
            ErrorKind::Dispatch,
            ErrorKind::Convergence,
            ErrorKind::Dimension,
            ErrorKind::Shape,
            ErrorKind::Index,
            ErrorKind::Value,
            ErrorKind::Type,
            ErrorKind::NotImplemented,
            ErrorKind::Implementation,
            ErrorKind::Memory,
            ErrorKind::Allocation,
            ErrorKind::Config,
            ErrorKind::InvalidArgument,
            ErrorKind::InvalidInput,
            ErrorKind::Permission,
            ErrorKind::Validation,
            ErrorKind::InvalidState,
            ErrorKind::Jit,
            ErrorKind::Json,
            ErrorKind::Io,
            ErrorKind::Scheduler,
            ErrorKind::Timeout,
            ErrorKind::Compression,
            ErrorKind::InvalidShape,
            ErrorKind::Device,
            ErrorKind::Mutex,
            ErrorKind::Thread,
            ErrorKind::Stream,
            ErrorKind::EndOfStream,
            ErrorKind::Resource,
            ErrorKind::Communication,
            ErrorKind::Security,
        ];
        let names: HashSet<&str> = kinds.iter().map(|k| k.as_str()).collect();
        assert_eq!(
            names.len(),
            kinds.len(),
            "Each ErrorKind must have a unique as_str()"
        );
    }

    #[test]
    fn test_context_accessor() {
        let err = CoreError::TimeoutError(
            ErrorContext::new("timed out")
                .with_location(super::super::error::ErrorLocation::new("test.rs", 42)),
        );
        let ctx = err.context();
        assert_eq!(ctx.message, "timed out");
        assert!(ctx.location.is_some());
    }

    #[test]
    fn test_with_backtrace_does_not_panic() {
        let ctx = ErrorContext::new("test").with_backtrace();
        // Should not panic; message should contain at least original text
        assert!(ctx.message.contains("test"));
    }
}
