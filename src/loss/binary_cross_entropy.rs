use crate::loss::Loss;
use ndarray::{Array, Array2, Ix, Ix2};

pub struct BinaryCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for BinaryCrossEntropy {
    #[inline(always)]
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl BinaryCrossEntropy {
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix2> for BinaryCrossEntropy {
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix2>) -> f64 {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);
        let y_true = &y_true.mapv(|x| x as f64);
        let sample_losses =
            -y_true * clamped_y_pred.ln() - (1.0 - y_true) * (1.0 - clamped_y_pred).ln();

        sample_losses.mean().unwrap()
    }

    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix2>) -> Array2<f64> {
        let clamped_y_pred = &y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        let y_true = &y_true.mapv(|x| x as f64);
        let gradient = -(y_true / clamped_y_pred) + (1.0 - y_true) / (1.0 - clamped_y_pred);

        &gradient / gradient.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_arr_eq_approx, assert_eq_approx};
    use ndarray::array;

    #[test]
    fn test_binary_cross_entropy() {
        let y_pred = array![[0.1, 0.5, 0.7], [0.25, 0.33, 0.75]];
        let y_true = array![[1, 0, 1], [0, 0, 1]];

        let loss = BinaryCrossEntropy::default().calculate(&y_pred, &y_true);

        assert_eq_approx!(loss, 0.7213748214989018);
    }

    #[test]
    fn test_binary_cross_entropy_backward() {
        let y_pred = array![[0.1, 0.5, 0.7], [0.25, 0.33, 0.75]];
        let y_true = array![[1, 0, 1], [0, 0, 1]];

        let gradient = BinaryCrossEntropy::default().backwards(&y_pred, &y_true);

        let expected = array![
            [-1.6666666666666667, 0.3333333333333333, -0.2380952380952381],
            [0.2222222222222222, 0.24875621890547264, -0.2222222222222222]
        ];

        assert_arr_eq_approx!(expected, gradient);
    }
}
