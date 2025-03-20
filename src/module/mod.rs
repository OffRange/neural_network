use crate::module::layers::TrainableLayer;
use crate::State;
use ndarray::{Array, Dimension};

pub mod activations;
pub mod layers;

pub trait Module {
    type Input: Dimension;
    type Output: Dimension;

    /// Performs the forward pass for this module.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an `Array2<f64>` representing the input data where each row is a sample.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the output of this module.
    fn forward(&mut self, input: &Array<f64, Self::Input>) -> Array<f64, Self::Output>;

    /// Performs the backward pass for this module.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the module's output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the module's input.
    fn backward(&mut self, d_value: &Array<f64, Self::Output>) -> Array<f64, Self::Input>;

    fn update_state(&mut self, _state: State) {}

    fn as_trainable_mut(
        &mut self,
    ) -> Option<&mut dyn TrainableLayer<Input = Self::Input, Output = Self::Output>> {
        None
    }
}
