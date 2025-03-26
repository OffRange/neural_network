use crate::doc_cross_entropy;
use crate::loss::Loss;
use crate::utils::ToOneHot;
use ndarray::{Array, Array2, Ix, Ix1, Ix2};

/// Computes the categorical cross entropy loss and its gradient.
///
/// This loss function is typically used for multi-class classification tasks where the true labels
/// are one hot encoded. It first clamps the predicted probabilities to avoid numerical
/// instability (e.g. caused by taking the logarithm of zero) when computing the loss and its derivative.
pub struct CategoricalCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for CategoricalCrossEntropy {
    #[doc = doc_cross_entropy!(default CategoricalCrossEntropy 1e-7)]
    #[inline(always)]
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl CategoricalCrossEntropy {
    #[doc = doc_cross_entropy!(new CategoricalCrossEntropy)]
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix> for CategoricalCrossEntropy {
    type PredDim = Ix2;
    type TargetDim = Ix2;

    /// Calculates the categorical cross entropy loss between the predicted probabilities and the true labels.
    ///
    /// The predicted probabilities are first clamped within the range
    /// `[clamp_epsilon, 1.0 - clamp_epsilon]` to prevent numerical instability when computing logarithms.
    /// The loss is computed as the negative log-likelihood and then averaged over all samples.
    ///
    /// The categorical cross entropy loss is defined as:
    ///
    /// ```math
    /// Lᵢ = L(yᵢ, ŷᵢ) = ∑ yᵢ,ⱼ * -ln(ŷᵢ,ⱼ)
    ///                j
    /// ```
    ///
    /// Where `yᵢ` is the true label for sample `i`, `ŷᵢ` is the predicted probability for sample `i`,
    /// and `j` iterates over all classes. As we expect the true labels to be one-hot encoded, the sum
    /// simplifies to the true label at index k times the logarithm of the predicted probability
    /// at index k for the true label.
    ///
    /// ```math
    /// Lᵢ = -yᵢ,ₖ * ln(ŷᵢ,ₖ)
    /// ```
    ///
    /// The final loss is the mean of all sample losses:
    ///
    /// ```math
    /// _   1   N
    /// L = ―   ∑ Lᵢ
    ///     N  i=1
    /// ```
    ///
    /// Where N is the number of samples.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted probabilities.
    /// * `y_true` - A 2D array of true labels, where each element is cast to f64.
    ///
    /// # Returns
    ///
    /// The mean categorical cross entropy loss as a `f64` value.
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

    /// Computes the gradient of the categorical cross entropy loss with respect to the predictions.
    ///
    /// The method first clamps the predicted probabilities to avoid division by zero
    /// when calculating the gradient. The gradient is then computed for each element
    /// by applying the derivative of the categorical cross entropy loss function, and is normalized
    /// by the total number of elements.
    ///
    /// The derivative of the categorical cross entropy loss function is:
    ///
    /// ```math
    /// ∂L(y, ŷ)     -y
    /// ――――― = ――――
    ///    ∂ŷ       ŷ * N
    /// ```
    ///
    /// Where `y` is the true label, `ŷ` is the predicted probability, and `N` is the number of samples.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted probabilities.
    /// * `y_true` - A 2D array of true labels, where each element is cast to f64.
    ///
    /// # Returns
    ///
    /// A 2D array containing the gradient of the loss with respect to each predicted probability.
    fn backwards(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<Ix, Self::TargetDim>,
    ) -> Array2<f64> {
        let samples = y_pred.nrows() as f64;

        let gradient = -y_true.mapv(|x| x as f64) / y_pred;
        gradient / samples
    }
}

/// Computes the categorical cross entropy loss along with its gradient.
///
/// This loss function is ideal for multi-class classification tasks where the ground truth labels are provided
/// as sparse indices (i.e., each label directly indicates the correct class). To ensure numerical stability,
/// the predicted probabilities are clamped to avoid issues like computing the logarithm of zero.
pub struct SparseCategoricalCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for SparseCategoricalCrossEntropy {
    #[doc = doc_cross_entropy!(default SparseCategoricalCrossEntropy 1e-7)]
    #[inline(always)]
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl SparseCategoricalCrossEntropy {
    #[doc = doc_cross_entropy!(new SparseCategoricalCrossEntropy)]
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix> for SparseCategoricalCrossEntropy {
    type PredDim = Ix2;
    type TargetDim = Ix1;


    /// Calculates the (sparse) categorical cross entropy loss between the predicted probabilities and the true labels.
    ///
    /// The predicted probabilities are first clamped within the range
    /// `[clamp_epsilon, 1.0 - clamp_epsilon]` to prevent numerical instability when computing logarithms.
    /// The loss is computed as the negative log-likelihood and then averaged over all samples.
    ///
    /// The loss function is absiclly the same as the one in [CategoricalCrossEntropy::calculate], but
    /// the true labels are provided as sparse indices instead of one-hot encoded vectors.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted probabilities.
    /// * `y_true` - A 1D array of true labels, where each element is cast to `usize`.
    ///
    /// # Returns
    ///
    /// The mean (sparse) categorical cross entropy loss as a `f64` value.
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

    /// Computes the gradient of the (sparse) categorical cross entropy loss with respect to the predictions.
    ///
    /// The method first clamps the predicted probabilities to avoid division by zero
    /// when calculating the gradient. The gradient is then computed for each element
    /// by applying the derivative of the (sparse) categorical cross entropy loss function, and is normalized
    /// by the total number of elements.
    ///
    /// The derivative is the same as in [CategoricalCrossEntropy::backwards], but the true labels are provided
    /// as sparse indices instead of one-hot encoded vectors.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted probabilities.
    /// * `y_true` - A 1D array of true labels, where each element is cast to `usize`.
    ///
    /// # Returns
    ///
    /// A 2D array containing the gradient of the loss with respect to each predicted probability.
    ///
    /// # Also see
    ///
    /// [CategoricalCrossEntropy::backwards]
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
