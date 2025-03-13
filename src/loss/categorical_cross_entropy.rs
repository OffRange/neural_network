use crate::loss::Loss;
use crate::utils::ToOneHot;
use ndarray::{Array, Array2, Ix, Ix1, Ix2};

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

impl Loss<Ix1, Ix> for CategoricalCrossEntropy {
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix1>) -> f64 {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        Array::from_shape_fn(clamped_y_pred.nrows(), |x| {
            -clamped_y_pred[[x, y_true[x]]].ln()
        })
        .mean()
        .unwrap()
    }

    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix1>) -> Array2<f64> {
        let one_hot = y_true.to_one_hot(y_pred.ncols());
        <Self as Loss<Ix2, Ix>>::backwards(self, y_pred, &one_hot)
    }
}

impl Loss<Ix2, Ix> for CategoricalCrossEntropy {
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix2>) -> f64 {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        -(y_true.mapv(|x| x as f64) * clamped_y_pred)
            .sum_axis(ndarray::Axis(1))
            .mapv(f64::ln)
            .mean()
            .unwrap() // Per-Sample Loss
    }

    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix2>) -> Array2<f64> {
        let samples = y_pred.nrows() as f64;

        let gradient = -y_true.mapv(|x| x as f64) / y_pred;
        gradient / samples // Normalize the gradient, this helps the optimizers
    }
}

#[cfg(test)]
mod tests {
    use super::{CategoricalCrossEntropy, Loss};
    use crate::{assert_arr_eq_approx, assert_eq_approx};
    use ndarray::array;

    #[test]
    fn test_categorical_cross_entropy() {
        let y_pred = array![[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08],];

        let y_true_one_hot = array![[1, 0, 0], [0, 1, 0], [0, 1, 0],];

        let y_true = array![0, 1, 1];

        let loss_one_hot = CategoricalCrossEntropy::default().calculate(&y_pred, &y_true_one_hot);
        let loss_sparse = CategoricalCrossEntropy::default().calculate(&y_pred, &y_true);

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
            CategoricalCrossEntropy::default().backwards(&y_pred, &y_true_sparse);

        let expected = array![
            [-1. / 2.1, 0.0, 0.0],
            [0.0, -2. / 3., 0.0],
            [0.0, -1. / 2.7, 0.0],
        ];

        assert_arr_eq_approx!(loss_backward_one_hot, expected);
        assert_arr_eq_approx!(loss_backward_sparse, expected);
    }
}
