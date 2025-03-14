use crate::State;
use ndarray::Array2;

pub mod activations;
pub mod layers;

pub trait Module {
    /// Performs the forward pass for this module.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an `Array2<f64>` representing the input data where each row is a sample.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the output of this module.
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;

    /// Performs the backward pass for this module.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the module's output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the module's input.
    fn backward(&mut self, d_value: &Array2<f64>) -> Array2<f64>;

    fn update_state(&mut self, state: State);
}
