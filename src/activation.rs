use ndarray::Array2;

pub trait ActivationFn: Default {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64>;
}

#[derive(Default)]
pub struct ReLU;

impl ActivationFn for ReLU {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.map(|x| x.max(0.0))
    }
}

pub struct LeakyReLU {
    alpha: f64,
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
        }
    }
}

impl ActivationFn for LeakyReLU {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.map(|&x| if x >= 0.0 { x } else { x * self.alpha })
    }
}

#[derive(Default)]
pub struct Softmax;

impl ActivationFn for Softmax {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let max = input.map_axis(ndarray::Axis(1), |row| {
            row.iter().copied().reduce(f64::max).unwrap()
        }).insert_axis(ndarray::Axis(1));

        let exp = (input - max).exp();
        let sum = exp.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        exp / sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_arr_eq_approx, assert_eq_approx};
    use ndarray::Array1;

    #[test]
    fn test_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, 0.0], [0.0, 4.0]];
        let output = ReLU::default().forward(&input);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, -0.02], [-0.03, 4.0]];
        let output = LeakyReLU::new(0.01).forward(&input);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_softmax() {
        let array = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 4.0],
        ];

        let expected_output = ndarray::array![
            [0.0900305731703804, 0.2447284710547976, 0.6652409557748218],
            [0.2119415576170854, 0.5761168847658291, 0.2119415576170854],
        ];

        let output = Softmax::default().forward(&array);
        let sum = output.sum_axis(ndarray::Axis(1));

        assert_arr_eq_approx!(output, expected_output);
        assert_arr_eq_approx!(sum, Array1::<f64>::ones(sum.len()));
    }
}