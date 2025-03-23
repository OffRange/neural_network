mod binary_cross_entropy;
mod categorical_cross_entropy;
mod mean_absolute_error;
mod mean_squared_error;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;
pub use mean_absolute_error::*;
pub use mean_squared_error::*;

use crate::metric::{Metric, MetricValue};
use ndarray::{Array, Dimension};

pub trait Loss<T> {
    type PredDim: Dimension;
    type TargetDim: Dimension;

    fn calculate(
        &self,
        y_pred: &Array<f64, <Self as Loss<T>>::PredDim>,
        y_true: &Array<T, <Self as Loss<T>>::TargetDim>,
    ) -> f64;

    fn backwards(
        &self,
        y_pred: &Array<f64, <Self as Loss<T>>::PredDim>,
        y_true: &Array<T, <Self as Loss<T>>::TargetDim>,
    ) -> Array<f64, <Self as Loss<T>>::PredDim>;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>().rsplit(':').next().unwrap()
    }
}

impl<T, L> Metric<T> for L
where
    L: Loss<T>,
{
    type PredDim = L::PredDim;
    type TargetDim = L::TargetDim;

    fn evaluate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<T, Self::TargetDim>,
    ) -> MetricValue {
        let loss = self.calculate(y_pred, y_true);
        MetricValue::new(self.name(), loss)
    }

    fn name(&self) -> &'static str {
        <Self as Loss<T>>::name(self)
    }
}
