use crate::layer::Layer;

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
            let w_momentum = self.momentum * layer.weight_momentum()
                - current_lr * layer.weights_gradient();

            let b_momentum = self.momentum * layer.bias_momentum()
                - current_lr * layer.biases_gradient();

            layer.weight_momentum_mut().assign(&w_momentum);
            layer.bias_momentum_mut().assign(&b_momentum);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initializer;
    use crate::layer::{Dense, Layer};
    use ndarray::array;

    #[test]
    fn test_sgd() {
        let mut layer = Dense::new::<initializer::He>(2, 2);
        layer.weights_mut().assign(&array![[0.1, 0.2], [0.3, 0.4]]);
        layer.biases_mut().assign(&array![0.1, 0.2]);

        let x = array![[1.0, 2.0]];
        layer.forward(&x);
        layer.backward(&x);

        let mut sgd = SGD::new(0.1, 0.9, 0.01);
        sgd.update(&mut layer);

        println!("{:#?}", layer);
    }
}