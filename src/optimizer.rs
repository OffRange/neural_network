use crate::layer::Layer;
use ndarray::{Array1, Array2};

pub trait Optimizer {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: Layer;
}

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: Layer,
    {
        let updater = |current_weights: &Array2<f64>, current_biases: &Array1<f64>, weights_gradient: &Array2<f64>, biases_gradient: &Array1<f64>| {
            (
                current_weights - self.lr * weights_gradient,
                current_biases - self.lr * biases_gradient,
            )
        };

        layer.update_params(updater);
    }
}