mod binary_cross_entropy;
mod categorical_cross_entropy;
mod mean_absolute_error;
mod mean_squared_error;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;
pub use mean_absolute_error::*;
pub use mean_squared_error::*;

use ndarray::Array;

pub trait Loss<T, PredDim, TrueDim>
where
    PredDim: ndarray::Dimension,
    TrueDim: ndarray::Dimension,
{
    fn calculate(&self, y_pred: &Array<f64, PredDim>, y_true: &Array<T, TrueDim>) -> f64;
    fn backwards(
        &self,
        y_pred: &Array<f64, PredDim>,
        y_true: &Array<T, TrueDim>,
    ) -> Array<f64, PredDim>;
}
