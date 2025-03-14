mod dense;
mod dropout;

pub use dense::*;
pub use dropout::*;

use crate::Module;
use crate::regularizer::Regularizer;
use ndarray::{Array1, Array2, ArrayViewMut1, ArrayViewMut2, Ix1, Ix2};

pub trait TrainableLayer: Module {
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
