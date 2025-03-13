use crate::loss::Loss;
use ndarray::{Array, Array2, Ix, Ix2};

/// BinaryCrossEntropy computes the binary cross entropy loss and its gradient.
///
/// This loss function is typically used for binary classification tasks. It clamps
/// the predicted probabilities to avoid numerical issues (e.g., taking the logarithm
/// of zero) when computing the loss and its derivative.
pub struct BinaryCrossEntropy {
    clamp_epsilon: f64,
}

impl Default for BinaryCrossEntropy {
    /// Creates a default instance of BinaryCrossEntropy.
    ///
    /// The default value for `clamp_epsilon` is set to 1e-7.
    #[inline(always)]
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl BinaryCrossEntropy {
    /// Creates a new instance of BinaryCrossEntropy with a specified clamp epsilon.
    ///
    /// # Arguments
    ///
    /// * `clamp_epsilon` - A small value to clamp predicted probabilities and avoid
    ///   numerical instability (e.g., taking the logarithm of zero).
    ///
    /// # Returns
    ///
    /// A new `BinaryCrossEntropy` instance.
    #[inline(always)]
    pub fn new(clamp_epsilon: f64) -> Self {
        Self { clamp_epsilon }
    }
}

impl Loss<Ix2, Ix> for BinaryCrossEntropy {
    /// Calculates the binary cross entropy loss between the predicted probabilities and the true labels.
    ///
    /// The predicted probabilities are first clamped within the range
    /// `[clamp_epsilon, 1.0 - clamp_epsilon]` to prevent numerical instability when computing logarithms.
    /// The loss is computed as the negative log-likelihood and then averaged over all samples.
    ///
    /// To introduce the function let's start with the Bernoulli distribution:
    ///
    /// ```math
    /// p(y | ŷ) = ŷ^y * (1 - ŷ)^(1 - y)
    /// ```
    ///
    /// Where y is the true binary label (y ∈ {0, 1}) and ŷ ∈ [0, 1] is the predicted probability.
    /// This means that if y = 1, the probability is ŷ, and if y = 0, the probability is 1 - ŷ.
    /// Taking the negated logarithm of the Bernoulli distribution gives us - after applying some
    /// logarithmic rules - the binary cross entropy loss:
    ///
    /// ```math
    /// L(y, ŷ) = -ln(p(y | ŷ)) = -y * ln(ŷ) - (1 - y) * ln(1 - ŷ)
    /// ```
    ///
    /// Again, this means that if y = 1, the loss is -log(ŷ), and if y = 0, the loss is -log(1 - ŷ).
    /// To ensure numerical stability, the predicted probabilities are clamped to the mentioned
    /// range to prevent taking the logarithm of zero.
    ///
    /// As this is a per-sample (for each data point) loss, the final loss is the mean of all sample losses.
    ///
    /// ```math
    /// _   1   N
    /// L = ―   ∑ L(yᵢ, ŷᵢ)
    ///     N  i=1
    /// ```
    ///
    /// Where N is the number of data points - basically the size of the passed matrix.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted probabilities.
    /// * `y_true` - A 2D array of true binary labels (0 or 1), where each element is cast to f64.
    ///
    /// # Returns
    ///
    /// The mean binary cross entropy loss as a `f64` value.
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<Ix, Ix2>) -> f64 {
        let clamped_y_pred = y_pred.clamp(self.clamp_epsilon, 1.0 - self.clamp_epsilon);
        let y_true = &y_true.mapv(|x| x as f64);
        let sample_losses =
            -y_true * clamped_y_pred.ln() - (1.0 - y_true) * (1.0 - clamped_y_pred).ln();

        sample_losses.mean().unwrap()
    }

    /// Computes the gradient of the binary cross entropy loss with respect to the predictions.
    ///
    /// The method first clamps the predicted probabilities to avoid division by zero
    /// when calculating the gradient. The gradient is then computed for each element
    /// by applying the derivative of the binary cross entropy loss function, and is normalized
    /// by the total number of elements.
    ///
    /// The derivative of the binary cross entropy loss function is:
    ///
    /// ```math
    /// ∂L(y, ŷ) / ∂ŷ = -y / ŷ + (1 - y) / (1 - ŷ)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted probabilities.
    /// * `y_true` - A 2D array of true binary labels (0 or 1), where each element is cast to f64.
    ///
    /// # Returns
    ///
    /// A 2D array containing the gradient of the loss with respect to each predicted probability.
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
