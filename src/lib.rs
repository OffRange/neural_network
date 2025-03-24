/*!
This crate is a personal, educational project designed to help both the author and others learn
about machine learning and deep learning. It is not intended for production use, and the
implementations may not follow industry best practices or be optimized for performance.

**Disclaimer:** The code is experimental and provided "as-is" without any warranty. Users are
encouraged to use it as a learning resource and modify or extend it as needed.
*/
#![doc = "## Mnist Example\n```"]
#![doc = include_str!("../examples/mnist.rs")]
#![doc = "\n```"]

#[cfg(feature = "blas")]
extern crate blas_src;

mod assert;
pub mod data;
pub mod initializer;
pub mod loss;
pub mod metric;
pub mod model;
pub mod module;
pub mod optimizers;
pub mod regularizer;
pub mod state;
pub mod utils;
pub mod value;

pub use module::Module;
pub use state::State;
