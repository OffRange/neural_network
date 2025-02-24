use ndarray::{Array, Array1, Array2, Ix, Ix1, Ix2};

pub trait Loss<D>
where
    D: ndarray::Dimension,
{
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, D>) -> Array1<f64>;

    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, D>) -> f64 {
        self.forward(y_pred, y_true).mean().unwrap()
    }
}

pub struct CategoricalCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for CategoricalCrossEntropy {
    #[inline(always)]
    fn default() -> Self {
        Self::new(f64::EPSILON)
    }
}

impl CategoricalCrossEntropy {
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix1> for CategoricalCrossEntropy {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix1>) -> Array1<f64> {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        Array::from_shape_fn(clamped_y_pred.nrows(), |x| {
            -clamped_y_pred[[x, y_true[x]]].ln()
        })
    }
}

impl Loss<Ix2> for CategoricalCrossEntropy {
    fn forward(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix2>) -> Array1<f64> {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);

        -(y_true.mapv(|x| x as f64) * clamped_y_pred)
            .sum_axis(ndarray::Axis(1))
            .mapv(f64::ln)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_eq_approx;
    use crate::loss::{CategoricalCrossEntropy, Loss};
    use ndarray::array;

    #[test]
    fn test_categorical_cross_entropy() {
        let y_pred = array![
            [0.7, 0.1, 0.2],
            [0.1, 0.5, 0.4],
            [0.02, 0.9, 0.08],
        ];

        let y_true_one_hot = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ].mapv(|x| x as usize);

        let y_true = array![0, 1, 1];

        let loss_one_hot_enc = CategoricalCrossEntropy::default().calculate(&y_pred, &y_true_one_hot);
        let loss = CategoricalCrossEntropy::default().calculate(&y_pred, &y_true);

        assert_eq_approx!(loss, 0.38506088005216804);
        assert_eq_approx!(loss_one_hot_enc, 0.38506088005216804);
    }
}