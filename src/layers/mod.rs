mod dense;
mod dropout;

pub use dense::*;
pub use dropout::*;

use crate::regularizer::Regularizer;
use crate::state::State;
use ndarray::{Array1, Array2, ArrayViewMut1, ArrayViewMut2, Ix1, Ix2};

pub trait Layer {
    /// Performs the forward pass for the layers.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an `Array2<f64>` representing the input data where each row is a sample.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the output of the layers.
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;

    /// Performs the backward pass for the layers.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the layers's output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the layers's input.
    fn backward(&mut self, d_value: &Array2<f64>) -> Array2<f64>;

    fn update_state(&mut self, state: State);
}

pub trait TrainableLayer: Layer {
    // Cache accessors
    // Used by optimizers to update the weights and biases
    fn weight_momentum(&self) -> &Array2<f64>;
    fn weight_momentum_mut(&mut self) -> &mut Array2<f64>;
    fn bias_momentum(&self) -> &Array1<f64>;
    fn bias_momentum_mut(&mut self) -> &mut Array1<f64>;

    fn weight_cache(&self) -> &Array2<f64>;
    fn weight_cache_mut(&mut self) -> &mut Array2<f64>;
    fn bias_cache(&self) -> &Array1<f64>;
    fn bias_cache_mut(&mut self) -> &mut Array1<f64>;

    // Parameter accessors
    fn weights(&self) -> &Array2<f64>;
    fn biases(&self) -> &Array1<f64>;
    fn weights_mut(&mut self) -> ArrayViewMut2<f64>;
    fn biases_mut(&mut self) -> ArrayViewMut1<f64>;
    fn weights_gradient(&self) -> &Array2<f64>;
    fn biases_gradient(&self) -> &Array1<f64>;

    fn kernel_regularizer(&self) -> Option<&dyn Regularizer<Ix2>>;
    fn bias_regularizer(&self) -> Option<&dyn Regularizer<Ix1>>;

    /// Returns the regularization losses for the kernel and bias weights.
    fn regularization_losses(&self) -> (f64, f64) {
        let kernel_loss = if let Some(reg) = self.kernel_regularizer() {
            reg.compute(self.weights())
        } else {
            0.0
        };

        let bias_loss = if let Some(reg) = self.bias_regularizer() {
            reg.compute(self.biases())
        } else {
            0.0
        };

        (kernel_loss, bias_loss)
    }
}
