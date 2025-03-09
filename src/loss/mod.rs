mod binary_cross_entropy;
mod categorical_cross_entropy;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;

use ndarray::{Array, Array2, Ix};

pub trait Loss<D>
where
    D: ndarray::Dimension,
{
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, D>) -> f64;
    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, D>) -> Array2<f64>;
}
