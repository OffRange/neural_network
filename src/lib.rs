#[cfg(feature = "blas")]
extern crate blas_src;

pub mod activations;
mod assert;
pub mod data;
pub mod initializer;
pub mod layers;
pub mod loss;
pub mod metric;
pub mod optimizers;
pub mod regularizer;
pub mod state;
pub mod utils;

pub use state::State;
