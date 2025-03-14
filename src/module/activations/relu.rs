use crate::{Module, State};
use ndarray::Array2;

#[derive(Default)]
pub struct ReLU {
    input: Option<Array2<f64>>,
}

impl Module for ReLU {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, d_values: &Array2<f64>) -> Array2<f64> {
        self.input
            .as_ref()
            .expect("input was not set. Please run the forward pass first.")
            .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
            * d_values
    }

    fn update_state(&mut self, _state: State) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, 0.0], [0.0, 4.0]];
        let output = ReLU::default().forward(&input);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_relu_backward() {
        let input = ndarray::array![[1., 2., -3., -4.], [2., -7., -1., 3.], [-1., 2., 5., -1.]];

        let dvalues = ndarray::array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]];

        let expected = ndarray::array![[1., 2., 0., 0.], [5., 0., 0., 8.], [0., 10., 11., 0.]];

        let mut relu = ReLU::default();
        let _ = relu.forward(&input);
        let d = relu.backward(&dvalues);

        assert_eq!(d, expected);
    }
}
