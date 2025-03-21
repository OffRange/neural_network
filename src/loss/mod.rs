mod binary_cross_entropy;
mod categorical_cross_entropy;
mod mean_absolute_error;
mod mean_squared_error;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;
pub use mean_absolute_error::*;
pub use mean_squared_error::*;

use ndarray::{Array, Dimension};

pub trait Loss<T> {
    type PredDim: Dimension;
    type TargetDim: Dimension;

    fn calculate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<T, Self::TargetDim>,
    ) -> f64;

    fn backwards(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<T, Self::TargetDim>,
    ) -> Array<f64, Self::PredDim>;
}
