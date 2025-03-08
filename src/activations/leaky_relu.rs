use crate::activations::ActivationFn;
use ndarray::Array2;

pub struct LeakyReLU {
    alpha: f64,
    input: Option<Array2<f64>>,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            input: None,
        }
    }
}

impl ActivationFn for LeakyReLU {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        input.map(|&x| if x >= 0.0 { x } else { x * self.alpha })
    }

    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64> {
        self.input
            .as_ref()
            .expect("input was not set. Please run the forward pass first.")
            .mapv(|x| if x >= 0.0 { 1.0 } else { self.alpha }) * d_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, -0.02], [-0.03, 4.0]];
        let output = LeakyReLU::new(0.01).forward(&input);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_leaky_relu_backwards() {
        let input = ndarray::array![
            [1., 2., -3., -4.],
            [2., -7., -1., 3.],
            [-1., 2., 5., -1.]
        ];

        let dvalues = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.]
        ];

        let expected = ndarray::array![
            [1., 2., 0.03, 0.04],
            [5., 0.06, 0.07, 8.],
            [0.09, 10., 11., 0.12]
        ];

        let mut leaky_relu = LeakyReLU::new(0.01);
        let _ = leaky_relu.forward(&input);
        let d = leaky_relu.backward(&dvalues);

        assert_eq!(d, expected);
    }
}