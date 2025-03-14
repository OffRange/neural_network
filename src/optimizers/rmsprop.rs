use crate::module::layers::TrainableLayer;
use crate::optimizers::Optimizer;
use std::ops::SubAssign;

pub struct RMSProp {
    lr: f64,
    decay: f64,
    epsilon: f64,
    roh: f64,
    iteration: usize,
    current_lr: f64,
}

impl RMSProp {
    pub fn new(lr: f64, decay: f64, epsilon: f64, roh: f64) -> Self {
        Self {
            lr,
            decay,
            epsilon,
            roh,
            iteration: 0,
            current_lr: lr,
        }
    }
}

impl Optimizer for RMSProp {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: TrainableLayer,
    {
        // Weights
        {
            let g = layer.weights_gradient().clone();
            let cache = layer.weight_cache_mut();
            cache.zip_mut_with(&g, |c, g| *c = self.roh * *c + (1. - self.roh) * g.powi(2));

            let update = self.current_lr * g / (cache.sqrt() + self.epsilon);
            layer.weights_mut().sub_assign(&update);
        }

        // Biases
        {
            let g = layer.biases_gradient().clone();
            let cache = layer.bias_cache_mut();
            cache.zip_mut_with(&g, |c, g| *c = self.roh * *c + (1. - self.roh) * g.powi(2));

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
mod test {
    use super::*;
    use crate::assert_arr_eq_approx;
    use crate::optimizers::test::prepared_layer;
    use ndarray::array;

    #[test]
    fn test_rmsprop() {
        let mut layer = prepared_layer();

        let mut rmsprop = RMSProp::new(0.001, 0., 1e-7, 0.9);
        rmsprop.pre_update();
        rmsprop.update(&mut layer);

        let expected_weights = array![
            [0.09683772333983132, 0.19683772283983156],
            [0.2968377228398315, 0.39683772258983163],
        ];
        let expected_biases = array![0.09683772333983132, 0.19683772283983156];
        let expected_weights_cache = array![
            [0.09999999999999998, 0.3999999999999999],
            [0.3999999999999999, 1.5999999999999996]
        ];
        let expected_biases_cache = array![0.1, 0.4];

        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }
}
