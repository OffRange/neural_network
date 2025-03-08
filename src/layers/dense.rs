use crate::initializer::Initializer;
use crate::layers::{Layer, TrainableLayer};
use crate::regularizer::Regularizer;
use crate::state::State;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;

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

    kernel_regularizer: Option<Box<dyn Regularizer<Ix2>>>,
    bias_regularizer: Option<Box<dyn Regularizer<Ix1>>>,
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
            kernel_regularizer: Default::default(),
            bias_regularizer: Default::default(),
        }
    }
}

impl Dense {
    pub fn new_with_regularizers<I>(n_input: usize, n_neurons: usize, kernel_regularizer: Option<Box<dyn Regularizer<Ix2>>>, bias_regularizer: Option<Box<dyn Regularizer<Ix1>>>) -> Self
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
            kernel_regularizer,
            bias_regularizer,
        }
    }

    pub fn new<I>(n_input: usize, n_neurons: usize) -> Self
    where
        I: Initializer,
    {
        Self::new_with_regularizers::<I>(n_input, n_neurons, None, None)
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, value: &Array2<f64>) -> Array2<f64> {
        let input = self.input
            .as_ref()
            .expect("input was not set. Please run the forward pass first.");

        let d_weights = input.t().dot(value);
        if let Some(reg) = self.kernel_regularizer() {
            let d_weights = d_weights + &reg.gradient(&self.weights);
            self.weights_gradient.assign(&d_weights);
        } else {
            self.weights_gradient.assign(&d_weights);
        }

        let d_biases = value.sum_axis(ndarray::Axis(0));
        if let Some(reg) = self.bias_regularizer() {
            let d_biases = d_biases + &reg.gradient(&self.biases);
            self.biases_gradient.assign(&d_biases);
        } else {
            self.biases_gradient.assign(&d_biases);
        }

        value.dot(&self.weights.t())
    }

    fn update_state(&mut self, _state: State) {}
}

impl TrainableLayer for Dense {
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

    fn kernel_regularizer(&self) -> Option<&Box<dyn Regularizer<Ix2>>> {
        self.kernel_regularizer.as_ref()
    }

    fn bias_regularizer(&self) -> Option<&Box<dyn Regularizer<Ix1>>> {
        self.bias_regularizer.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_approx;
    use crate::initializer::test::ConstantInitializer;
    use ndarray::{array, Array2};

    #[test]
    fn test_dense_new_shapes_and_values() {
        let n_input = 3;
        let n_neurons = 2;
        let layer = Dense::new::<ConstantInitializer<1>>(n_input, n_neurons);

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
        let mut layer = Dense::new::<ConstantInitializer<1>>(3, 2);

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
        let mut layer = Dense::new::<ConstantInitializer<1>>(3, 2);

        // Incorrect input shape: Only 2 columns instead of 3.
        let input = array![[1.0, 2.0]];
        let _ = layer.forward(&input); // Should panic
    }
}