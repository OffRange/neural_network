use crate::expect;
use crate::initializer::Initializer;
use ndarray::{Array1, Array2, ArrayViewMut1, ArrayViewMut2};
use ndarray_rand::RandomExt;
use std::fmt::Debug;

pub trait Layer {
    fn new<I>(n_input: usize, n_neurons: usize) -> Self
    where
        I: Initializer;

    /// Performs the forward pass for the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an `Array2<f64>` representing the input data where each row is a sample.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the output of the layer.
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;

    /// Performs the backward pass for the layer.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the layer's output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the layer's input.
    fn backward(&mut self, d_value: &Array2<f64>) -> Array2<f64>;

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
}

#[derive(Debug)]
pub struct Dense {
    /// A matrix of shape (n_in, n_neurons). This allows us to skip the transpose operation during#
    /// the forward pass.
    weights: Array2<f64>,

    /// A vector of shape (n_neurons,).
    biases: Array1<f64>,
    input: Option<Array2<f64>>,

    weights_gradient: Array2<f64>,
    biases_gradient: Array1<f64>,

    weight_momentum: Array2<f64>,
    bias_momentum: Array1<f64>,

    weight_cache: Array2<f64>,
    bias_cache: Array1<f64>,
}

#[cfg(debug_assertions)]
impl Dense {
    pub fn new_with_weights_and_biases(weights: Array2<f64>, biases: Array1<f64>) -> Self {
        Self {
            weights: weights.clone(),
            biases: biases.clone(),
            input: None,
            weights_gradient: Array2::zeros(weights.raw_dim()),
            biases_gradient: Array1::zeros(biases.raw_dim()),
            weight_momentum: Array2::zeros(weights.raw_dim()),
            bias_momentum: Array1::zeros(biases.raw_dim()),
            weight_cache: Array2::zeros(weights.raw_dim()),
            bias_cache: Array1::zeros(biases.raw_dim()),
        }
    }
}

impl Layer for Dense {
    fn new<I>(n_input: usize, n_neurons: usize) -> Self
    where
        I: Initializer,
    {
        let initializer = I::new(n_input, n_neurons);

        let weights = Array2::random((n_input, n_neurons), initializer.dist());
        let biases = Array1::random(n_neurons, initializer.dist());

        Self {
            weights,
            biases,
            input: None,
            weights_gradient: Array2::zeros((n_input, n_neurons)),
            biases_gradient: Array1::zeros(n_neurons),
            weight_momentum: Array2::zeros((n_input, n_neurons)),
            bias_momentum: Array1::zeros(n_neurons),
            weight_cache: Array2::zeros((n_input, n_neurons)),
            bias_cache: Array1::zeros(n_neurons),
        }
    }

    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, value: &Array2<f64>) -> Array2<f64> {
        self.weights_gradient.assign(&expect!(self.input.as_ref()).t().dot(value));
        self.biases_gradient.assign(&value.sum_axis(ndarray::Axis(0)));

        value.dot(&self.weights.t())
    }

    fn weight_momentum(&self) -> &Array2<f64> {
        &self.weight_momentum
    }

    fn weight_momentum_mut(&mut self) -> &mut Array2<f64> {
        &mut self.weight_momentum
    }

    fn bias_momentum(&self) -> &Array1<f64> {
        &self.bias_momentum
    }

    fn bias_momentum_mut(&mut self) -> &mut Array1<f64> {
        &mut self.bias_momentum
    }

    fn weight_cache(&self) -> &Array2<f64> {
        &self.weight_cache
    }

    fn weight_cache_mut(&mut self) -> &mut Array2<f64> {
        &mut self.weight_cache
    }

    fn bias_cache(&self) -> &Array1<f64> {
        &self.bias_cache
    }

    fn bias_cache_mut(&mut self) -> &mut Array1<f64> {
        &mut self.bias_cache
    }

    fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    fn biases(&self) -> &Array1<f64> {
        &self.biases
    }


    fn weights_mut(&mut self) -> ArrayViewMut2<f64> {
        self.weights.view_mut()
    }

    fn biases_mut(&mut self) -> ArrayViewMut1<f64> {
        self.biases.view_mut()
    }

    fn weights_gradient(&self) -> &Array2<f64> {
        &self.weights_gradient
    }

    fn biases_gradient(&self) -> &Array1<f64> {
        &self.biases_gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_approx;
    use ndarray::{array, Array2};
    use ndarray_rand::rand::prelude::*;
    use ndarray_rand::rand_distr::Distribution;

    struct ConstantDistribution {
        constant: f64,
    }

    impl Distribution<f64> for ConstantDistribution {
        fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
            self.constant
        }
    }

    struct ConstantInitializer {
        constant: f64,
    }

    impl Initializer for ConstantInitializer {
        fn new(_fan_in: usize, _fan_out: usize) -> Self {
            Self { constant: 1.0 }
        }

        fn dist(&self) -> impl Distribution<f64> {
            ConstantDistribution {
                constant: self.constant,
            }
        }
    }

    #[test]
    fn test_dense_new_shapes_and_values() {
        let n_input = 3;
        let n_neurons = 2;
        let layer = Dense::new::<ConstantInitializer>(n_input, n_neurons);

        assert_eq!(layer.weights.shape(), &[n_input, n_neurons]);
        assert_eq!(layer.biases.shape(), &[n_neurons]);

        // Since the constant initializer always returns 1.0,
        // every element in weights and biases should equal 1.0.
        for &value in layer.weights.iter() {
            assert_eq_approx!(value, 1.0, "Expected weight to be 1.0, got {}", value);
        }
        for &value in layer.biases.iter() {
            assert_eq_approx!(value, 1.0, "Expected bias to be 1.0, got {}", value);
        }
    }

    #[test]
    fn test_backward() {
        let mut layer = Dense::new_with_weights_and_biases(
            array![[-0.00177312,  0.01083391],
                            [ 0.00998164, -0.0024269 ],
                            [-0.00253938, -0.00447975]
            ],
            array![0.0, 0.0],
        );

        let input: Array2<f64> = array![[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0]];

        let d_values: Array2<f64> = array![[1.0, 2.0],
                                          [3.0, 4.0]];

        layer.input = Some(input);
        let a = layer.backward(&d_values);
        println!("{:?}", layer.biases_gradient);
    }

    #[test]
    fn test_dense_forward() {
        let mut layer = Dense::new::<ConstantInitializer>(3, 2);

        let input: Array2<f64> = array![[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0]];

        // For each neuron, the output is:
        // dot(input_row, weights_column) + bias
        // Given weights are all 1's and bias is 1,
        // For the first row: (1+2+3) + 1 = 7, for both neurons.
        // For the second row: (4+5+6) + 1 = 16, for both neurons.
        let expected: Array2<f64> = array![[7.0, 7.0],
                                           [16.0, 16.0]];

        let output = layer.forward(&input);
        assert_eq!(output, expected, "The forward pass did not produce the expected output.");
    }


    /// This test checks that if we pass an input with the wrong dimensions,
    /// the underlying matrix multiplication panics.
    #[test]
    #[should_panic]
    fn test_forward_dimension_mismatch() {
        let mut layer = Dense::new::<ConstantInitializer>(3, 2);

        // Incorrect input shape: Only 2 columns instead of 3.
        let input = array![[1.0, 2.0]];
        let _ = layer.forward(&input); // Should panic
    }
}

