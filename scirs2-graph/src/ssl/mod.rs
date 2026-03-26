//! Graph Self-Supervised Learning (SSL) methods.
//!
//! Provides contrastive learning and masked autoencoder approaches for
//! learning graph representations without labels.
//!
//! | Sub-module | Method | Reference |
//! |---|---|---|
//! | [`contrastive`] | GraphCL, SimGRACE, NT-Xent loss | You et al. 2020; Xia 2022 |
//! | [`masked_autoencoder`] | GraphMAE with SCE loss | Hou et al. 2022 |

pub mod contrastive;
pub mod masked_autoencoder;
/// Graph pre-training strategies: node masking, graph-context contrastive, attribute reconstruction.
pub mod pretrain;

// Contrastive learning
pub use contrastive::{
    augment_edges, augment_features, nt_xent_loss, simgrace_perturb, GraphClConfig, ProjectionHead,
};

// Masked autoencoder
pub use masked_autoencoder::{GraphMae, GraphMaeConfig};

// Pre-training strategies
pub use pretrain::{
    infonce_loss, AttrReconConfig, AttrReconConfig as AttributeReconConfig,
    AttributeReconstructionObjective, GraphContextConfig, GraphContextPretrainer,
    NodeMaskingConfig, NodeMaskingPretrainer,
};
