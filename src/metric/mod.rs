mod multi_class_accuracy;
mod regression_accuracy;

use ndarray::{Array, ArrayView, Dimension};

pub use multi_class_accuracy::*;
pub use regression_accuracy::*;

pub trait Metric<A> {
    type PredDim: Dimension;
    type TargetDim: Dimension;

    fn evaluate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<A, Self::TargetDim>,
    ) -> f64;
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
    D: Dimension,
{
    data: ArrayView<'a, f64, D>,
    ddof: f64,
    epsilon: f64,
}

impl<D> Tolerance for StdTolerance<'_, D>
where
    D: Dimension,
{
    fn tolerance(&self) -> f64 {
        self.data.std(self.ddof) / self.epsilon
    }
}

impl<'a, D> StdTolerance<'a, D>
where
    D: Dimension,
{
    pub fn new(data: ArrayView<'a, f64, D>, ddof: f64, epsilon: f64) -> Self {
        Self {
            data,
            ddof,
            epsilon,
        }
    }
}
