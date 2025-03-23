mod multi_class_accuracy;
mod regression_accuracy;

use crate::value::Value;
pub use multi_class_accuracy::*;
use ndarray::{Array, ArrayView, Dimension};
pub use regression_accuracy::*;

#[derive(Debug, Default, Clone)]
pub struct MetricValue {
    name: &'static str,
    value: f64,
}

impl MetricValue {
    pub fn new(name: &'static str, value: f64) -> Self {
        Self { name, value }
    }
}

impl Value for MetricValue {
    fn value(&self) -> f64 {
        self.value
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

pub trait Metric<A> {
    type PredDim: Dimension;
    type TargetDim: Dimension;

    fn evaluate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<A, Self::TargetDim>,
    ) -> MetricValue;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
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
