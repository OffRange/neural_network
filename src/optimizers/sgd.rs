use crate::module::layers::TrainableLayer;
use crate::optimizers::Optimizer;

pub struct SGD {
    lr: f64,
    momentum: f64,
    decay: f64,
    iteration: usize,
    current_lr: f64,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, decay: f64) -> Self {
        Self {
            lr,
            momentum,
            decay,
            iteration: 0,
            current_lr: lr,
        }
    }
}

impl Optimizer for SGD {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: TrainableLayer,
    {
        if self.momentum != 0.0 {
            let w_momentum =
                self.momentum * layer.weight_cache() - self.current_lr * layer.weights_gradient();

            let b_momentum =
                self.momentum * layer.bias_cache() - self.current_lr * layer.biases_gradient();

            layer.weight_cache_mut().assign(&w_momentum);
            layer.bias_cache_mut().assign(&b_momentum);

            layer
                .weights_mut()
                .zip_mut_with(&w_momentum, |w, m| *w += m);
            layer.biases_mut().zip_mut_with(&b_momentum, |w, m| *w += m);

            return;
        }

        let g = layer.weights_gradient().clone();
        layer
            .weights_mut()
            .zip_mut_with(&g, |w, g| *w -= self.current_lr * g);
        let g = layer.biases_gradient().clone();
        layer
            .biases_mut()
            .zip_mut_with(&g, |b, g| *b -= self.current_lr * g);
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
    fn test_sgd() {
        let mut layer = prepared_layer();

        let mut sgd = SGD::new(0.1, 0.9, 0.01);
        sgd.pre_update();
        sgd.update(&mut layer);

        let expected_weights = array![[0., 0.], [0.1, 0.],];
        let expected_biases = array![0., 0.];
        let expected_weights_cache = array![[-0.1, -0.2], [-0.2, -0.4],];
        let expected_biases_cache = array![-0.1, -0.2];

        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }

    #[test]
    fn test_sdg_no_momentum() {
        let mut layer = prepared_layer();
        let mut sdg = SGD::new(0.1, 0., 0.01);
        sdg.pre_update();
        sdg.update(&mut layer);

        let expected_weights = array![[0., 0.], [0.1, 0.],];

        let expected_biases = array![0., 0.];

        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
    }
}
