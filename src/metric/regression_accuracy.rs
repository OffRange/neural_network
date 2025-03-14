use crate::metric::{Metric, Tolerance};
use ndarray::{Array, Array2, Ix2};

pub struct RegressionAccuracy<T>
where
    T: Tolerance,
{
    tolerance: T,
}

impl<T> RegressionAccuracy<T>
where
    T: Tolerance,
{
    pub fn new(tolerance: T) -> Self {
        Self { tolerance }
    }
}

impl Default for RegressionAccuracy<f64> {
    fn default() -> Self {
        Self { tolerance: 0.01 }
    }
}

impl<T> Metric<f64, Ix2> for RegressionAccuracy<T>
where
    T: Tolerance,
{
    fn evaluate(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> f64 {
        let diff = y_pred - y_true;
        let diff = diff.mapv(|x| (x.abs() < self.tolerance.tolerance()) as u8 as f64);

        diff.mean().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_approx;
    use ndarray::array;

    #[test]
    fn test_regression_accuracy() {
        let y_pred = array![[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]];

        let y_true = array![
            [1.0, 0.21, 0.12],
            [0.489, 0.08915, 0.6],
            [0.018, 0.9872, 0.08]
        ];

        let result = RegressionAccuracy::default().evaluate(&y_pred, &y_true);

        assert_eq_approx!(result, 1. / 3.);
    }
}
