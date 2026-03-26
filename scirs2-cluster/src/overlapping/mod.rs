//! Overlapping community detection algorithms.
//!
//! This module implements algorithms that detect communities where nodes may
//! belong to multiple communities simultaneously:
//!
//! - **BigClam**: Community Affiliation Graph Model via NMF of affiliation matrix
//! - **DEMON**: Local overlapping community detection via ego-network label propagation
//! - **Link Communities**: Edge partitioning to infer node overlaps
//! - **Evaluation**: Overlapping NMI, Omega index, F1, and coverage metrics

pub mod bigclam;
pub mod demon;
pub mod evaluation;
pub mod link_communities;

pub use bigclam::{BigClam, BigClamConfig, BigClamInit, MembershipMatrix};
pub use demon::{Demon, DemonConfig};
pub use evaluation::{coverage, omega_index, overlap_f1, overlapping_nmi};
pub use link_communities::{EdgeCommunity, LinkCommunities, LinkCommunityConfig};
