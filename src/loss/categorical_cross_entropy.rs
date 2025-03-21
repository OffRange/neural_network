use crate::loss::Loss;
use crate::utils::ToOneHot;
use ndarray::{Array, Array2, Dimension, Ix, Ix1, Ix2};

pub struct CategoricalCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for CategoricalCrossEntropy {
    #[inline(always)]
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl CategoricalCrossEntropy {
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix> for CategoricalCrossEntropy {
    type PredDim = Ix2;
    type TargetDim = Ix2;

    fn calculate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<Ix, Self::TargetDim>,
    ) -> f64 {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        -(y_true.mapv(|x| x as f64) * clamped_y_pred)
            .sum_axis(ndarray::Axis(1))
            .mapv(f64::ln)
            .mean()
            .unwrap() // Per-Sample Loss
    }

    fn backwards(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<Ix, Self::TargetDim>,
    ) -> Array2<f64> {
        let samples = y_pred.nrows() as f64;

        let gradient = -y_true.mapv(|x| x as f64) / y_pred;
        gradient / samples // Normalize the gradient, this helps the optimizers
    }
}

pub struct SparseCategoricalCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for SparseCategoricalCrossEntropy {
    #[inline(always)]
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl SparseCategoricalCrossEntropy {
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix> for SparseCategoricalCrossEntropy {
    type PredDim = Ix2;
    type TargetDim = Ix1;

    fn calculate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<Ix, Self::TargetDim>,
    ) -> f64 {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        Array::from_shape_fn(clamped_y_pred.nrows(), |x| {
            -clamped_y_pred[[x, y_true[x]]].ln()
        })
        .mean()
        .unwrap()
    }

    fn backwards(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<Ix, Self::TargetDim>,
    ) -> Array2<f64> {
        let one_hot = y_true.to_one_hot(y_pred.ncols());
        CategoricalCrossEntropy::new(self.clamp_epsilon).backwards(y_pred, &one_hot)
    }
}

#[cfg(test)]
mod tests {
    use super::{CategoricalCrossEntropy, Loss};
    use crate::loss::SparseCategoricalCrossEntropy;
    use crate::{assert_arr_eq_approx, assert_eq_approx};
    use ndarray::array;

    #[test]
    fn test_categorical_cross_entropy() {
        let y_pred = array![[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08],];

        let y_true_one_hot = array![[1, 0, 0], [0, 1, 0], [0, 1, 0],];

        let y_true = array![0, 1, 1];

        let loss_one_hot = CategoricalCrossEntropy::default().calculate(&y_pred, &y_true_one_hot);
        let loss_sparse = SparseCategoricalCrossEntropy::default().calculate(&y_pred, &y_true);

        assert_eq_approx!(loss_sparse, 0.38506088005216804);
        assert_eq_approx!(loss_one_hot, 0.38506088005216804);
    }

    #[test]
    fn test_categorical_cross_entropy_backward() {
        let y_pred = array![[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08],];

        let y_true_one_hot = array![[1, 0, 0], [0, 1, 0], [0, 1, 0],];

        let y_true_sparse = array![0, 1, 1];

        let loss_backward_one_hot =
            CategoricalCrossEntropy::default().backwards(&y_pred, &y_true_one_hot);
        let loss_backward_sparse =
            SparseCategoricalCrossEntropy::default().backwards(&y_pred, &y_true_sparse);

        let expected = array![
            [-1. / 2.1, 0.0, 0.0],
            [0.0, -2. / 3., 0.0],
            [0.0, -1. / 2.7, 0.0],
        ];

        assert_arr_eq_approx!(loss_backward_one_hot, expected);
        assert_arr_eq_approx!(loss_backward_sparse, expected);
    }
}
