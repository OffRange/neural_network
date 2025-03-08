use ndarray::{Array, Axis, Dimension, RemoveAxis};
use std::fmt::Debug;

pub trait Regularizer<D>: Debug
where
    D: Dimension,
{
    /// Computes the regularization term for parameters of a layers.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A reference to an `Array2<f64>` representing a set of parameters of a layers.
    ///
    /// # Returns
    ///
    /// * A scalar `f64` representing the regularization term.
    fn compute(&self, parameters: &Array<f64, D>) -> f64;

    /// Computes the gradient of the regularization term with respect to a set of parameters of a layers.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A reference to an `Array2<f64>` representing a set of parameters of a layers.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the regularization term with respect to the parameters.
    fn gradient(&self, parameters: &Array<f64, D>) -> Array<f64, D>;
}

#[macro_export]
macro_rules! reg_structure {
    ($(#[$meta:meta])* $vis:vis struct $reg_name:ident { $($fields:ident: $ty:ty)* }) => {
        $(#[$meta])*
        #[derive(Debug)]
        $vis struct $reg_name {
            $(
                $fields: $ty,
            )*
        }
    };

    ($(#[$meta:meta])* $vis:vis struct $reg_name:ident { $($fields:ident: $ty:ty $(= $init:tt)?)* }) => {
        $(#[$meta])*
        #[derive(Debug)]
        $vis struct $reg_name {
            $(
                $fields: $ty,
            )*
        }

        impl Default for $reg_name {
            fn default() -> Self {
                Self {
                    $(
                        $fields: $crate::reg_structure!(@maybe_default $($init)?),
                    )*
                }
            }
        }
    };

    (@maybe_default $init:expr) => {
        $init
    };

    (@maybe_default) => {
        Default::default()
    };
}

reg_structure! {
    /// L1 Regularization (Lasso Regularization)
    ///
    /// Applies a penalty equal to the absolute values of the weights. Unlike L2 regularization,
    /// L1 tends to penalize small weights more aggressively, effectively driving many of them to zero.
    /// This behavior causes the model to become invariant to minor changes in the input,
    /// responding primarily to larger differences. Due to this characteristic,
    /// L1 regularization is rarely used alone and is typically combined with L2 regularization
    /// (as in Elastic Net) to balance sparsity with smooth weight decay.
    pub struct L1 {
        lambda: f64 = 0.01
    }
}

impl L1 {
    /// Creates a new L1 regularization term.
    ///
    /// # Arguments
    ///
    /// * `lambda` - A scalar `f64` representing the regularization strength.
    ///
    /// # Returns
    ///
    /// * An `L1` instance.
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not positive.
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Lambda must be positive");
        Self { lambda }
    }
}

impl<D> Regularizer<D> for L1
where
    D: Dimension,
{
    fn compute(&self, parameters: &Array<f64, D>) -> f64 {
        self.lambda * parameters.abs().sum()
    }

    fn gradient(&self, parameters: &Array<f64, D>) -> Array<f64, D> {
        self.lambda * parameters.mapv(|x| x.signum())
    }
}

reg_structure! {
    /// L2 Regularization (Ridge Regularization)
    ///
    /// Applies a penalty proportional to the square of each weight, scaled by a regularization factor.
    /// Because the penalty increases quadratically with the weight's magnitude, it effectively discourages
    /// large weight values. This regularization prevents the model from growing large parameters as it
    /// penalizes relatively large values, ensuring that the model maintains smoother and more stable
    /// weight distributions. L2 regularization helps prevent overfitting and is often used either on its own
    /// or in combination with L1 regularization (as in Elastic Net) for a balanced approach.
    pub struct L2 {
        lambda: f64 = 0.01
    }
}

//TODO enable usage in combination with L1 regularization and wise versa
impl L2 {
    /// Creates a new L2 regularization term with the given lambda.
    ///
    /// # Arguments
    ///
    /// * `lambda` - A scalar `f64` representing the regularization strength.
    ///
    /// # Returns
    ///
    /// * An `L2` instance.
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not positive.
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Lambda must be positive");
        Self { lambda }
    }
}

impl<D> Regularizer<D> for L2
where
    D: Dimension + RemoveAxis,
{
    fn compute(&self, parameters: &Array<f64, D>) -> f64 {
        self.lambda
            * parameters
                .powi(2)
                .mean_axis(Axis(0)) // Axis 0 is the batch axis (in a 2D array --> rows)
                .unwrap()
                .sum()
    }

    fn gradient(&self, parameters: &Array<f64, D>) -> Array<f64, D> {
        let batch_size = parameters.len_of(Axis(0)) as f64;
        2. * self.lambda * parameters / batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arr_eq_approx;
    use crate::initializer::test::ConstantInitializer;
    use crate::layers::{Dense, Layer, TrainableLayer};
    use ndarray::{Array1, Array2};

    #[test]
    fn test_l1_regularization() {
        let l1 = L1::new(0.1);
        let parameters = Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap();
        let expected_grad = Array2::from_shape_vec((2, 2), vec![0.1, 0.1, 0.1, 0.1]).unwrap();

        assert_eq!(l1.compute(&parameters), 1.);
        assert_eq!(l1.gradient(&parameters), expected_grad);
    }

    #[test]
    fn test_l2_regularization() {
        let l2 = L2::new(0.1);
        let parameters = Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 4.5]).unwrap();
        let expected_grad = Array2::from_shape_vec(
            (3, 2),
            vec![1. / 15., 2. / 15., 1. / 5., 4. / 15., 5. / 15., 3. / 10.],
        )
        .unwrap();

        assert_eq!(l2.compute(&parameters), 301. / 120.);
        assert_arr_eq_approx!(l2.gradient(&parameters), expected_grad);
    }

    #[test]
    fn test_with_layer() {
        let mut layer = Dense::new_with_regularizers::<ConstantInitializer<1>>(
            5,
            5,
            Some(Box::new(L1::new(0.01))),
            None,
        );
        let input = Array2::from_elem((5, 5), 2.);

        layer.biases_mut().assign(&Array1::zeros(5));

        let output = layer.forward(&input);

        let (kernel_loss, _) = layer.regularization_losses();
        let activity_loss = L2::new(0.01);

        assert_eq!(kernel_loss, 0.25);
        assert_eq!(activity_loss.compute(&output), 5.);
    }
}
