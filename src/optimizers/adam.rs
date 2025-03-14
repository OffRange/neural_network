use crate::module::layers::TrainableLayer;
use crate::optimizers::Optimizer;
use std::ops::SubAssign;

pub struct Adam {
    lr: f64,
    lr_decay: f64,
    epsilon: f64,
    beta1: f64,
    beta2: f64,
    iteration: usize,
    current_lr: f64,
}

impl Adam {
    pub fn new(lr: f64, lr_decay: f64, epsilon: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            lr,
            lr_decay,
            epsilon,
            beta1,
            beta2,
            iteration: 0,
            current_lr: lr,
        }
    }
}

impl Optimizer for Adam {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: TrainableLayer,
    {
        // Weights
        {
            let gradient = layer.weights_gradient().clone();
            let momentum = layer.weight_momentum_mut();
            momentum.zip_mut_with(&gradient, |m, g| {
                *m = self.beta1 * *m + (1. - self.beta1) * g;
            });

            let momentum_corrected =
                momentum.map(|m| m / (1. - self.beta1.powi(self.iteration as i32)));

            let cache = layer.weight_cache_mut();
            cache.zip_mut_with(&gradient, |c, g| {
                *c = self.beta2 * *c + (1. - self.beta2) * g.powi(2);
            });

            let cache_corrected = cache.map(|c| c / (1. - self.beta2.powi(self.iteration as i32)));

            let update =
                self.current_lr * momentum_corrected / (cache_corrected.sqrt() + self.epsilon);
            layer.weights_mut().sub_assign(&update);
        }

        // Biases
        {
            let gradient = layer.biases_gradient().clone();
            let momentum = layer.bias_momentum_mut();
            momentum.zip_mut_with(&gradient, |m, g| {
                *m = self.beta1 * *m + (1. - self.beta1) * g;
            });

            let momentum_corrected =
                momentum.map(|m| m / (1. - self.beta1.powi(self.iteration as i32)));

            let cache = layer.bias_cache_mut();
            cache.zip_mut_with(&gradient, |c, g| {
                *c = self.beta2 * *c + (1. - self.beta2) * g.powi(2);
            });

            let cache_corrected = cache.map(|c| c / (1. - self.beta2.powi(self.iteration as i32)));

            let update =
                self.current_lr * momentum_corrected / (cache_corrected.sqrt() + self.epsilon);
            layer.biases_mut().sub_assign(&update);
        }
    }

    fn learning_rate(&self) -> f64 {
        self.current_lr
    }

    fn pre_update(&mut self) {
        self.current_lr = self.lr * (1.0 / (1.0 + self.lr_decay * self.iteration as f64));
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
    fn test_adam() {
        let mut layer = prepared_layer();

        let mut adam = Adam::new(0.001, 0., 1e-7, 0.9, 0.999);
        adam.pre_update();
        adam.update(&mut layer);

        let expected_weights = array![
            [0.0990000001, 0.19900000005000001],
            [0.29900000005, 0.399000000025],
        ];
        let expected_biases = array![0.0990000001, 0.19900000005000001];
        let expected_weights_momentum = array![
            [0.09999999999999998, 0.19999999999999996],
            [0.19999999999999996, 0.3999999999999999],
        ];
        let expected_biases_momentum = array![0.09999999999999998, 0.19999999999999996];
        let expected_weights_cache = array![
            [0.0010000000000000009, 0.0040000000000000036],
            [0.0040000000000000036, 0.016000000000000014]
        ];
        let expected_biases_cache = array![0.0010000000000000009, 0.0040000000000000036];

        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_momentum(), expected_weights_momentum);
        assert_arr_eq_approx!(layer.bias_momentum(), expected_biases_momentum);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }
}
