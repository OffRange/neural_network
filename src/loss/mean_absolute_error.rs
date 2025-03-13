use crate::loss::Loss;
use ndarray::{Array, Array2, Ix2};

#[derive(Default)]
pub struct MeanAbsoluteError;

impl Loss<Ix2, f64> for MeanAbsoluteError {
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> f64 {
        (y_true - y_pred).abs().mean().unwrap()
    }

    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> Array2<f64> {
        let diff = y_true - y_pred;
        let diff = diff.mapv(|x| {
            if x > 0. {
                1.
            } else if x < 0. {
                -1.
            } else {
                0.
            }
        });
        diff / y_true.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arr_eq_approx;
    use ndarray::array;

    #[test]
    fn test_mean_absolute_error_calculate() {
        let y_pred = array![[1., 2., 3.], [4., 5., 6.]];
        let y_true = array![[1., 1.5, 3.], [4., 4.5, 6.5]];
        let loss = MeanAbsoluteError.calculate(&y_pred, &y_true);
        assert_eq!(loss, 0.25);
    }

    #[test]
    fn test_mean_absolute_error_backwards() {
        let y_pred = array![[1., 2., 3.], [4., 5., 6.]];
        let y_true = array![[1., 1.5, 3.], [4., 4.5, 6.5]];
        let d = MeanAbsoluteError.backwards(&y_pred, &y_true);

        let expected = array![[0., -1. / 6., 0.], [0., -1. / 6., 1. / 6.]];

        assert_arr_eq_approx!(d, expected);
    }
}
