//! Persistent and specialised collection types for scirs2-core.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`bit_vec`] | Compact bit-vector |
//! | [`flat_map`] | Flat (sorted-vec) ordered map |
//! | [`tiny_vec`] | Small-vector optimised for ≤ 4 elements |
//! | [`rrb_vec`] | Persistent radix-balanced tree vector |

pub mod bit_vec;
pub mod flat_map;
pub mod rrb_vec;
pub mod tiny_vec;

pub use bit_vec::BitVec;
pub use flat_map::FlatMap;
pub use rrb_vec::RrbVec;
pub use tiny_vec::TinyVec;
