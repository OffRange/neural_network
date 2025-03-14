mod multi_class_accuracy;
mod regression_accuracy;

use ndarray::{Array, Array2, ArrayView};

pub use multi_class_accuracy::*;
pub use regression_accuracy::*;

pub trait Metric<A, D>
where
    D: ndarray::Dimension,
{
    fn evaluate(&self, y_pred: &Array2<f64>, y_true: &Array<A, D>) -> f64;
}

pub trait Tolerance {
    fn tolerance(&self) -> f64;
}

impl Tolerance for f64 {
    fn tolerance(&self) -> f64 {
        *self
    }
}

pub struct StdTolerance<'a, D>
where
    D: ndarray::Dimension,
{
    data: ArrayView<'a, f64, D>,
    ddof: f64,
    epsilon: f64,
}

impl<D> Tolerance for StdTolerance<'_, D>
where
    D: ndarray::Dimension,
{
    fn tolerance(&self) -> f64 {
        self.data.std(self.ddof) / self.epsilon
    }
}

impl<'a, D> StdTolerance<'a, D>
where
    D: ndarray::Dimension,
{
    pub fn new(data: ArrayView<'a, f64, D>, ddof: f64, epsilon: f64) -> Self {
        Self {
            data,
            ddof,
            epsilon,
        }
    }
}
