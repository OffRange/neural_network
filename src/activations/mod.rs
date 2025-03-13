mod leaky_relu;
mod linear;
mod relu;
mod sigmoid;
mod softmax;

pub use leaky_relu::*;
pub use linear::*;
pub use relu::*;
pub use sigmoid::*;
pub use softmax::*;

use ndarray::Array2;

// TODO generalize to support multiple dimensions
pub trait ActivationFn: Default {
    /// Performs the forward pass for the activations function.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an `Array2<f64>` representing the input data.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the activated output.
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;

    /// Performs the backward pass for the activations function.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the activations output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the activations input.
    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64>;
}
