use crate::utils::argmax;
use ndarray::{Array, Array2, ArrayView, Ix, Ix1, Ix2};

pub trait Metric<A, D>
where
    D: ndarray::Dimension,
{
    fn evaluate(&self, y_pred: &Array2<f64>, y_true: &Array<A, D>) -> f64;
}

#[derive(Default)]
pub struct MultiClassAccuracy;

impl Metric<f64, Ix2> for MultiClassAccuracy {
    fn evaluate(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> f64 {
        let y_true = argmax(y_true);
        Self::evaluate(self, y_pred, &y_true)
    }
}

impl Metric<Ix, Ix1> for MultiClassAccuracy {
    fn evaluate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix1>) -> f64 {
        let y_pred = argmax(y_pred);
        y_pred
            .iter()
            .zip(y_true.iter())
            .filter(|(pred, true_)| pred == true_)
            .count() as f64
            / y_pred.len() as f64
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
    fn test_multiclass_accuracy() {
        let y_pred = array![[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]];

        let y_true = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];

        let y_true_scalar = argmax(&y_true); // Equivalent to [0, 1, 1]

        let result_one_hot_enc = MultiClassAccuracy.evaluate(&y_pred, &y_true);
        let result = MultiClassAccuracy.evaluate(&y_pred, &y_true_scalar);

        assert_eq_approx!(result, 2. / 3.);
        assert_eq_approx!(result_one_hot_enc, 2. / 3.);
    }
}
