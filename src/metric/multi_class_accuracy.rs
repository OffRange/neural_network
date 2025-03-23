use crate::metric::{Metric, MetricValue};
use crate::utils::Argmax;
use ndarray::{Array, Axis, Ix, Ix1, Ix2};

#[derive(Default)]
pub struct MultiClassAccuracy;

impl Metric<f64> for MultiClassAccuracy {
    type PredDim = Ix2;
    type TargetDim = Ix2;

    fn evaluate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<f64, Self::TargetDim>,
    ) -> MetricValue {
        let y_true = y_true.argmax(Axis(1));
        Self::evaluate(self, y_pred, &y_true)
    }
}

impl Metric<Ix> for MultiClassAccuracy {
    type PredDim = Ix2;
    type TargetDim = Ix1;

    fn evaluate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<Ix, Self::TargetDim>,
    ) -> MetricValue {
        let y_pred = y_pred.argmax(Axis(1));
        let value = y_pred
            .iter()
            .zip(y_true.iter())
            .filter(|(pred, true_)| pred == true_)
            .count() as f64
            / y_pred.len() as f64;

        MetricValue::new(<Self as Metric<Ix>>::name(self), value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_approx;
    use crate::value::Value;
    use ndarray::array;

    #[test]
    fn test_multiclass_accuracy() {
        let y_pred = array![[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]];

        let y_true = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];

        let y_true_scalar = y_true.argmax(Axis(1)); // Equivalent to [0, 1, 1]

        let result_one_hot_enc = MultiClassAccuracy.evaluate(&y_pred, &y_true).value();
        let result = MultiClassAccuracy.evaluate(&y_pred, &y_true_scalar).value();

        assert_eq_approx!(result, 2. / 3.);
        assert_eq_approx!(result_one_hot_enc, 2. / 3.);
    }
}
