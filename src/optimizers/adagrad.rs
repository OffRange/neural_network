use crate::module::layers::TrainableLayer;
use crate::optimizers::Optimizer;
use std::ops::{Mul, SubAssign};

pub struct AdaGrad {
    lr: f64,
    decay: f64,
    epsilon: f64,
    iteration: usize,
    current_lr: f64,
}

impl AdaGrad {
    pub fn new(lr: f64, decay: f64, epsilon: f64) -> Self {
        Self {
            lr,
            decay,
            epsilon,
            iteration: 0,
            current_lr: lr,
        }
    }
}

impl Optimizer for AdaGrad {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: TrainableLayer,
    {
        // Weights
        {
            let g = layer.weights_gradient().clone();
            let cache = layer.weight_cache_mut();
            *cache += &g.powi(2);

            let update = g.mul(self.current_lr) / (cache.sqrt() + self.epsilon);
            layer.weights_mut().sub_assign(&update);
        }

        // Biases
        {
            let g = layer.biases_gradient().clone();
            let cache = layer.bias_cache_mut();
            *cache += &g.powi(2);

            let update = self.current_lr * g / (cache.sqrt() + self.epsilon);
            layer.biases_mut().sub_assign(&update);
        }
    }

    fn learning_rate(&self) -> f64 {
        self.current_lr
    }

    fn pre_update(&mut self) {
        self.current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration as f64));
        self.iteration += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arr_eq_approx;
    use crate::optimizers::test::prepared_layer;
    use ndarray::array;

    #[test]
    fn test_adagrad() {
        let mut layer = prepared_layer();

        let mut adagrad = AdaGrad::new(1., 0., 1e-7);
        adagrad.pre_update();
        adagrad.update(&mut layer);

        let expected_weights = array![
            [-0.89999990000001, -0.7999999500000026],
            [-0.6999999500000027, -0.5999999750000006],
        ];
        let expected_biases = array![-0.89999990000001, -0.7999999500000026];
        let expected_weights_cache = array![[1., 4.], [4., 16.],];
        let expected_biases_cache = array![1., 4.];

        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }
}
