mod binary_cross_entropy;
mod categorical_cross_entropy;
mod mean_squared_error;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;
pub use mean_squared_error::*;

use ndarray::{Array, Array2};

pub trait Loss<D, T>
where
    D: ndarray::Dimension,
{
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<T, D>) -> f64;
    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<T, D>) -> Array2<f64>;
}
