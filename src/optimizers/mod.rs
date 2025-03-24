mod adagrad;
mod adam;
mod rmsprop;
mod sgd;

pub use adagrad::*;
pub use adam::*;
pub use rmsprop::*;
pub use sgd::*;

use crate::module::layers::TrainableLayer;

pub trait Optimizer {
    fn update<L>(&mut self, layer: &mut L)
    where
        L: TrainableLayer + ?Sized;

    fn learning_rate(&self) -> f64;
    fn pre_update(&mut self);
}

#[doc(hidden)]
#[macro_export]
macro_rules! __optimizer_builder {
    (
        [$($meta:meta)* $vis:vis $builder_name:ident for $name:ident]
        [$($field:ident: $type:ty $(= $default:expr)?),*]
        [$($internal_field:ident: $internal_type:ty = $internal_default:tt),*]
    ) => {
        $(#[$meta])*
        $vis struct $builder_name {
            $($field: $type),*
        }

        impl Default for $builder_name {
            fn default() -> Self {
                Self {
                    $($field: $crate::__optimizer_builder!(@maybe_default $($default)?)),*
                }
            }
        }

        impl $builder_name {
            $(
                pub fn $field(mut self, $field: $type) -> Self {
                    self.$field = $field;
                    self
                }
            )*

            pub fn build(self) -> $name {
                $name::from(self)
            }
        }

        $vis struct $name {
            $($field: $type),*
            $(, $internal_field: $internal_type)*
        }

        impl $name {
            pub fn new($($field: $type),*) -> Self {
                Self {
                    $($field),*
                    $(, $internal_field: $crate::__optimizer_builder!(@maybe_default $internal_default))*
                }
            }
        }

        impl From<$builder_name> for $name {
            fn from(builder: $builder_name) -> Self {
                Self {
                    $($field: builder.$field),*
                    $(, $internal_field: $crate::__optimizer_builder!(@maybe_default builder $internal_default))*
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                $builder_name::default().build()
            }
        }
    };

    (@maybe_default $builder:ident $init:ident) => {
        $builder.$init
    };

    (@maybe_default $builder:ident $init:expr) => {
        $crate::__optimizer_builder!(@maybe_default $init)
    };

    (@maybe_default $init:expr) => {
        $init
    };

    (@maybe_default) => {
        Default::default()
    };
}

/// Generates a struct definition for optimizers with optional default field values and a builder
/// pattern for creating optimizer instances.
///
/// This macro automatically creates a builder struct with configurable hyper-parameters
/// for an optimizer type. It allows users to conveniently set options with default values,
/// while also supporting an optional internal block for managing non-exposed, derived, or
/// state-related fields. **Note:** This macro does not implement the [optimizer](Optimizer) trait for the
/// target structure.
///
/// # Example
///
/// ```rust
/// use neural_network::optimizer_builder;
///
/// optimizer_builder! {
///     pub struct AdamBuilder for Adam {
///         // Public hyper-parameters with their default values:
///         lr: f64 = 0.001,
///         lr_decay: f64 = 0.,
///         epsilon: f64 = 1e-7,
///         beta1: f64 = 0.9,
///         beta2: f64 = 0.999;
///
///         // Optional internal block for non-public fields:
///         // Useful for storing state, derived values, or caching that
///         // should not be exposed as part of the public API.
///         internal = {
///             iteration: usize = 0,
///             current_lr: f64 = lr // initialized by the `lr` hyper-parameter
///         }
///     }
/// }
/// ```
///
/// The optional `internal` block defines fields that are not exposed to the user,
/// which is ideal for internal state or values computed from user-defined parameters.
#[macro_export]
macro_rules! optimizer_builder {
    (
        $(#[$meta:meta])*
        $vis:vis struct $builder_name:ident for $name:ident {
            $($field:ident: $type:ty $(= $default:expr)?),* $(,)?
            $(; internal = {
                $($internal_field:ident: $internal_type:ty = $internal_default:tt),* $(,)?
            })?
        }
    ) => {
        $crate::__optimizer_builder! {
            [$(#[$meta])* $vis $builder_name for $name]
            [$($field: $type $(= $default)?),*]
            [$($($internal_field: $internal_type = $internal_default),*)?]
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::initializer;
    use crate::module::{layers::Dense, Module};
    use ndarray::array;

    pub(super) fn prepared_layer() -> Dense {
        let mut layer = Dense::new::<initializer::He>(2, 2);
        layer.weights_mut().assign(&array![[0.1, 0.2], [0.3, 0.4]]);
        layer.biases_mut().assign(&array![0.1, 0.2]);

        let x = array![[1.0, 2.0]];
        layer.forward(&x);
        layer.backward(&x);

        layer
    }
}
