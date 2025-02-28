use crate::layer::{Dense, Layer};
use std::ops::{AddAssign, Mul, SubAssign};

pub trait Optimizer {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: Layer;
}

pub struct SGD {
    lr: f64,
    momentum: f64,
    decay: f64,
    iteration: usize,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, decay: f64) -> Self {
        Self {
            lr,
            momentum,
            decay,
            iteration: 0,
        }
    }
}

impl Optimizer for SGD {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: Layer,
    {
        let current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration as f64));

        self.iteration += 1;

        if self.momentum != 0.0 {
            let w_momentum = self.momentum * layer.weight_cache()
                - current_lr * layer.weights_gradient();

            let b_momentum = self.momentum * layer.bias_cache()
                - current_lr * layer.biases_gradient();

            layer.weight_cache_mut().assign(&w_momentum);
            layer.bias_cache_mut().assign(&b_momentum);

            layer.weights_mut().zip_mut_with(&w_momentum, |w, m| *w += m);
            layer.biases_mut().zip_mut_with(&b_momentum, |w, m| *w += m);

            return;
        }

        let g = layer.weights_gradient().clone();
        layer.weights_mut().zip_mut_with(&g, |w, g| {
            *w -= current_lr * g
        });
        let g = layer.biases_gradient().clone();
        layer.biases_mut().zip_mut_with(&g, |b, g| {
            *b -= current_lr * g
        });
    }
}

pub struct AdaGrad {
    lr: f64,
    decay: f64,
    epsilon: f64,
    iteration: usize,
}

impl AdaGrad {
    pub fn new(lr: f64, decay: f64, epsilon: f64) -> Self {
        Self {
            lr,
            decay,
            epsilon,
            iteration: 0,
        }
    }
}

impl Optimizer for AdaGrad {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: Layer,
    {
        let current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration as f64));

        self.iteration += 1;


        // Weights
        {
            let g = layer.weights_gradient().clone();
            let cache = layer.weight_cache_mut();
            *cache += &g.powi(2);

            let update = g.mul(current_lr) / (cache.sqrt() + self.epsilon);
            layer.weights_mut().sub_assign(&update);
        }

        // Biases
        {
            let g = layer.biases_gradient().clone();
            let cache = layer.bias_cache_mut();
            *cache += &g.powi(2);

            let update = current_lr * g / (cache.sqrt() + self.epsilon);
            layer.biases_mut().sub_assign(&update);
        }
    }
}

pub struct RMSProp {
    lr: f64,
    decay: f64,
    epsilon: f64,
    roh: f64,
    iteration: usize,
}

impl RMSProp {
    pub fn new(lr: f64, decay: f64, epsilon: f64, roh: f64) -> Self {
        Self {
            lr,
            decay,
            epsilon,
            roh,
            iteration: 0,
        }
    }
}

impl Optimizer for RMSProp {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: Layer,
    {
        let current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration as f64));

        self.iteration += 1;


        // Weights
        {
            let g = layer.weights_gradient().clone();
            let cache = layer.weight_cache_mut();
            cache.zip_mut_with(&g, |c, g| *c = self.roh * *c + (1. - self.roh) * g.powi(2));


            let update = current_lr * g / (cache.sqrt() + self.epsilon);
            layer.weights_mut().sub_assign(&update);
        }

        // Biases
        {
            let g = layer.biases_gradient().clone();
            let cache = layer.bias_cache_mut();
            cache.zip_mut_with(&g, |c, g| *c = self.roh * *c + (1. - self.roh) * g.powi(2));


            let update = current_lr * g / (cache.sqrt() + self.epsilon);
            layer.biases_mut().sub_assign(&update);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_approx;
    use crate::layer::{Dense, Layer};
    use crate::{assert_arr_eq_approx, initializer};
    use ndarray::array;

    fn prepared_layer() -> Dense {
        let mut layer = Dense::new::<initializer::He>(2, 2);
        layer.weights_mut().assign(&array![[0.1, 0.2], [0.3, 0.4]]);
        layer.biases_mut().assign(&array![0.1, 0.2]);

        let x = array![[1.0, 2.0]];
        layer.forward(&x);
        layer.backward(&x);

        layer
    }

    #[test]
    fn test_sgd() {
        let mut layer = prepared_layer();

        let mut sgd = SGD::new(0.1, 0.9, 0.01);
        sgd.update(&mut layer);

        let expected_weights = array![
            [0., 0.],
            [0.1, 0.],
        ];
        let expected_biases = array![0., 0.];
        let expected_weights_cache = array![
            [-0.1, -0.2],
            [-0.2, -0.4],
        ];
        let expected_biases_cache = array![-0.1, -0.2];


        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }

    #[test]
    fn test_adagrad() {
        let mut layer = prepared_layer();

        let mut adagrad = AdaGrad::new(1., 0., 1e-7);
        adagrad.update(&mut layer);

        let expected_weights = array![
            [-0.89999990000001, -0.7999999500000026],
            [-0.6999999500000027, -0.5999999750000006],
        ];
        let expected_biases = array![-0.89999990000001, -0.7999999500000026];
        let expected_weights_cache = array![
            [1., 4.],
            [4., 16.],
        ];
        let expected_biases_cache = array![1., 4.];


        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }

    #[test]
    fn test_rmsprop() {
        let mut layer = prepared_layer();

        let mut rmsprop = RMSProp::new(0.001, 0., 1e-7, 0.9);
        rmsprop.update(&mut layer);

        let expected_weights = array![
            [0.09683772333983132, 0.19683772283983156],
            [0.2968377228398315, 0.39683772258983163],
        ];
        let expected_biases = array![0.09683772333983132, 0.19683772283983156];
        let expected_weights_cache = array![
            [0.09999999999999998, 0.3999999999999999 ],
            [0.3999999999999999, 1.5999999999999996 ]
        ];
        let expected_biases_cache = array![0.1, 0.4];


        assert_arr_eq_approx!(layer.weights(), expected_weights);
        assert_arr_eq_approx!(layer.biases(), expected_biases);
        assert_arr_eq_approx!(layer.weight_cache(), expected_weights_cache);
        assert_arr_eq_approx!(layer.bias_cache(), expected_biases_cache);
    }
}