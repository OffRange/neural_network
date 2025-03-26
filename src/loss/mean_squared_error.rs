use crate::loss::Loss;
use ndarray::{Array, Array2, Ix2};

/// Computes the mean squared error between predicted and true values.
///
/// This loss function is typically used for regression tasks and penalizes the squared difference
/// between predicted and true values (non-linear penalisation). It is typically used more often
/// than the mean absolute error loss function as it is more sensitive to outliers.
///
/// # Also see
///
/// * [MeanAbsoluteError](crate::loss::MeanAbsoluteError)
#[derive(Default)]
pub struct MeanSquaredError;

impl Loss<f64> for MeanSquaredError {
    type PredDim = Ix2;
    type TargetDim = Ix2;

    /// Calculates the mean squared error between the predicted and true values.
    ///
    /// The mean squared error is defined as:
    ///
    /// ```math
    /// L(y, ŷ) = (y - ŷ)^2
    /// ```
    ///
    /// Where `y` is the true label, and `ŷ` is the predicted value.
    /// The final loss is the mean of all sample losses.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted values.
    /// * `y_true` - A 2D array of true  labels, where each element is cast to f64.
    ///
    /// # Returns
    ///
    /// The avg mean absolute error loss as a `f64` values.
    fn calculate(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> f64 {
        (y_true - y_pred).pow2().mean().unwrap()
    }

    /// Computes the gradient of the mean squared error loss with respect to the predictions.
    ///
    /// The gradient is computed for each element by applying the derivative of the mean squared error loss function,
    /// and is normalized by the total number of elements.
    ///
    /// The derivative of the mean squared error loss function is:
    ///
    /// ```math
    /// ∂L(y, ŷ) / ∂ŷ  = 2 * (y - ŷ)
    /// ```
    ///
    /// Where `y` is the true label, and `ŷ` is the predicted value.
    ///
    ///
    /// # Arguments
    ///
    /// * `y_pred` - A 2D array of predicted values.
    /// * `y_true` - A 2D array of true labels, where each element is cast to f64.
    ///
    /// # Returns
    ///
    /// A 2D array containing the gradient of the loss with respect to each predicted value.
    fn backwards(&self, y_pred: &Array2<f64>, y_true: &Array<f64, Ix2>) -> Array2<f64> {
        -2.0 * (y_true - y_pred) / y_true.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arr_eq_approx;
    use ndarray::array;

    #[test]
    fn test_mean_squared_error_calculate() {
        let y_pred = array![[1., 2., 3.], [4., 5., 6.]];
        let y_true = array![[1., 1.5, 3.], [4., 4.5, 6.5]];
        let loss = MeanSquaredError.calculate(&y_pred, &y_true);
        assert_eq!(loss, 0.125);
    }

    #[test]
    fn test_mean_squared_error_backwards() {
        let y_pred = array![[1., 2., 3.], [4., 5., 6.]];
        let y_true = array![[1., 1.5, 3.], [4., 4.5, 6.5]];
        let d = MeanSquaredError.backwards(&y_pred, &y_true);

        let expected = array![[0., 1. / 6., 0.], [0., 1. / 6., -1. / 6.]];

        assert_arr_eq_approx!(d, expected);
    }
}
