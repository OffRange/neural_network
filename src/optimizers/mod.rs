mod adagrad;
mod adam;
mod rmsprop;
mod sgd;

pub use adagrad::*;
pub use adam::*;
pub use rmsprop::*;
pub use sgd::*;

use crate::module::layers::TrainableLayer;

pub trait Optimizer {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: TrainableLayer;

    fn learning_rate(&self) -> f64;
    fn pre_update(&mut self);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::initializer;
    use crate::module::{Module, layers::Dense};
    use ndarray::array;

    pub(super) fn prepared_layer() -> Dense {
        let mut layer = Dense::new::<initializer::He>(2, 2);
        layer.weights_mut().assign(&array![[0.1, 0.2], [0.3, 0.4]]);
        layer.biases_mut().assign(&array![0.1, 0.2]);

        let x = array![[1.0, 2.0]];
        layer.forward(&x);
        layer.backward(&x);

        layer
    }
}
