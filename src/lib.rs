#[cfg(feature = "blas")]
extern crate blas_src;

pub mod layers;
pub mod initializer;
pub mod data;
pub mod activations;
mod assert;
pub mod loss;
pub mod metric;
pub mod utils;
pub mod optimizers;
pub mod regularizer;
pub mod state;

pub use state::State;

