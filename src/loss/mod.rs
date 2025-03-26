/*! Provides the `Loss` trait and implementations for common loss functions. */

mod binary_cross_entropy;
mod categorical_cross_entropy;
mod mean_absolute_error;
mod mean_squared_error;
mod doc;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;
pub use mean_absolute_error::*;
pub use mean_squared_error::*;

use crate::metric::{Metric, MetricValue};
use ndarray::{Array, Dimension};

/// A trait defining loss functions for machine learning models.
///
/// Loss functions measure the discrepancy between predicted and true values,
/// providing both a scalar loss value and gradient information for backpropagation.
///
/// # Type Parameter
///
/// - `T`: The type of target values (often `f64` or an enum for classification)
///
/// # Example
///
/// ```rust
/// use ndarray::Array;
/// use neural_network::loss::Loss;
///
/// // Hypothetical custom loss implementation
/// struct CustomLoss;
///
/// impl Loss<f64> for CustomLoss {
///     type PredDim = ndarray::Ix2;  // 2D prediction array
///     type TargetDim = ndarray::Ix2;  // 2D target array
///
///     fn calculate(
///         &self,
///         y_pred: &Array<f64, Self::PredDim>,
///         y_true: &Array<f64, Self::TargetDim>,
///     ) -> f64 {
///         // Example loss calculation
///         y_pred.iter()
///             .zip(y_true.iter())
///             .map(|(p, t)| (p - t).powi(2))
///             .sum()
///     }
///
///     fn backwards(
///         &self,
///         y_pred: &Array<f64, Self::PredDim>,
///         y_true: &Array<f64, Self::TargetDim>,
///     ) -> Array<f64, Self::PredDim> {
///         // Example gradient calculation
///         y_pred - y_true
///     }
/// }
/// ```
pub trait Loss<T> {
    /// Dimensionality of the prediction array.
    ///
    /// Allows flexible support for different input shapes (e.g., batch predictions, single predictions).
    type PredDim: Dimension;

    /// Dimensionality of the target array.
    ///
    /// Allows flexible support for different target shapes.
    type TargetDim: Dimension;

    /// Calculate the loss between predicted and true values.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - Predicted values
    /// * `y_true` - True target values
    ///
    /// # Returns
    ///
    /// A scalar loss value representing the error.
    fn calculate(
        &self,
        y_pred: &Array<f64, <Self as Loss<T>>::PredDim>,
        y_true: &Array<T, <Self as Loss<T>>::TargetDim>,
    ) -> f64;

    /// Compute the gradient of the loss with respect to predictions.
    ///
    /// Used for backpropagation during model training.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - Predicted values
    /// * `y_true` - True target values
    ///
    /// # Returns
    ///
    /// An array of gradients with the same shape as predictions.
    fn backwards(
        &self,
        y_pred: &Array<f64, <Self as Loss<T>>::PredDim>,
        y_true: &Array<T, <Self as Loss<T>>::TargetDim>,
    ) -> Array<f64, <Self as Loss<T>>::PredDim>;

    /// Get the name of the loss function.
    ///
    /// # Returns
    ///
    /// A static string representing the loss function's name.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_network::loss::Loss;
    ///
    /// struct MockLoss;
    ///
    /// impl Loss<f64> for MockLoss {
    ///     type PredDim = ndarray::Ix2;
    ///     type TargetDim = ndarray::Ix2;
    ///     fn calculate(&self, y_pred: &ndarray::Array<f64, Self::PredDim>, y_true: &ndarray::Array<f64, Self::TargetDim>) -> f64 { unimplemented!() }
    ///     fn backwards(&self, y_pred: &ndarray::Array<f64, Self::PredDim>, y_true: &ndarray::Array<f64, Self::TargetDim>) -> ndarray::Array<f64, Self::PredDim> { unimplemented!() }
    /// }
    ///
    /// let mock_loss = MockLoss;
    /// assert_eq!(mock_loss.name(), "MockLoss");
    /// ```
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>().rsplit(':').next().unwrap()
    }
}

/// Implements the `Metric` trait for any loss function.
///
/// This allows loss functions to be used as evaluation metrics during model training or validation.
impl<T, L> Metric<T> for L
where
    L: Loss<T>,
{
    type PredDim = L::PredDim;
    type TargetDim = L::TargetDim;

    fn evaluate(
        &self,
        y_pred: &Array<f64, Self::PredDim>,
        y_true: &Array<T, Self::TargetDim>,
    ) -> MetricValue {
        let loss = self.calculate(y_pred, y_true);
        MetricValue::new(self.name(), loss)
    }

    fn name(&self) -> &'static str {
        <Self as Loss<T>>::name(self)
    }
}
