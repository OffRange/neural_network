use crate::loss::Loss;
use ndarray::{Array, Array2, Ix2};

#[derive(Default)]
pub struct MeanSquaredError;

impl Loss<Ix2, f64> for MeanSquaredError {
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> f64 {
        (y_true - y_pred).pow2().mean().unwrap()
    }

    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> Array2<f64> {
        -2.0 * (y_true - y_pred) / y_true.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arr_eq_approx;
    use ndarray::array;

    #[test]
    fn test_mean_squared_error_calculate() {
        let y_pred = array![[1., 2., 3.], [4., 5., 6.]];
        let y_true = array![[1., 1.5, 3.], [4., 4.5, 6.5]];
        let loss = MeanSquaredError.calculate(&y_pred, &y_true);
        assert_eq!(loss, 0.125);
    }

    #[test]
    fn test_mean_squared_error_backwards() {
        let y_pred = array![[1., 2., 3.], [4., 5., 6.]];
        let y_true = array![[1., 1.5, 3.], [4., 4.5, 6.5]];
        let d = MeanSquaredError.backwards(&y_pred, &y_true);

        let expected = array![[0., 1. / 6., 0.], [0., 1. / 6., -1. / 6.]];

        assert_arr_eq_approx!(d, expected);
    }
}
