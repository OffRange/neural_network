use crate::Module;
use crate::state::State;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Bernoulli;

// TODO write tests for the dropout layers
pub struct Dropout {
    keep_prob: f64,
    mask: Option<Array2<f64>>,
    state: State,
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Dropout {
    pub fn new(dropout_rate: f64) -> Self {
        assert!(
            dropout_rate > 0.0 && dropout_rate <= 1.0,
            "Dropout rate must be in the range (0, 1]"
        );
        Self {
            keep_prob: 1. - dropout_rate,
            mask: None,
            state: Default::default(),
        }
    }
}

impl Module for Dropout {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        if self.state == State::Evaluating {
            return input.clone();
        }

        let mask = Array2::<bool>::random(input.raw_dim(), Bernoulli::new(self.keep_prob).unwrap());
        self.mask = Some(mask.mapv(f64::from));

        // Scaled to ensure that the overall expected summ remains the same.
        input * self.mask.as_ref().unwrap() / self.keep_prob
    }

    fn backward(&mut self, d_value: &Array2<f64>) -> Array2<f64> {
        if self.state == State::Evaluating {
            return d_value.clone();
        }
        // If the Bernoulli output is a one, the forward function is basically the input to the
        // dropout layers itself (typically layers' output) divided by the keep_prob value. If that
        // Bernoulli output is a zero, the forward function is zero.
        //
        // Dropout(z) = z * mask / keep_prob
        // d(Dropout(z))/dz = mask / keep_prob
        // z represents the layers' output and obvious the input to the dropout function.
        d_value
            * self
                .mask
                .as_ref()
                .expect("mask was not set. Please run the forward pass first.")
            / self.keep_prob
    }

    fn update_state(&mut self, state: State) {
        self.state = state
    }
}
