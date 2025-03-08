use crate::activations::ActivationFn;
use ndarray::Array2;

#[derive(Default)]
pub struct Softmax {
    output: Option<Array2<f64>>,
}

impl ActivationFn for Softmax {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let max = input
            .map_axis(ndarray::Axis(1), |row| {
                row.iter().copied().reduce(f64::max).unwrap()
            })
            .insert_axis(ndarray::Axis(1));

        let exp = (input - max).exp();
        let sum = exp.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        let out = exp / sum;
        self.output = Some(out.clone());
        out
    }

    /// Performs the backward pass for the activations function.
    ///
    /// For a softmax output S, where `S[i][j]` is the probability of the j-th class for the i-th sample,
    /// the derivative with respect to the input z is given by:
    ///
    /// ```ignore
    /// ∂S[i][j] / ∂z[i][k] = S[i][j] * (δ[j][k] - S[i][k])
    /// ```
    ///
    /// Here, `δ[j][k]` is the Kronecker delta, which equals 1 if j = k and 0 otherwise.
    ///
    /// In matrix form, the Jacobian can be expressed as:
    ///
    /// ```ignore
    /// J = diag(S) - S · Sᵀ
    /// ```
    ///
    /// where:
    /// - diag(S) is a diagonal matrix with the softmax outputs along the diagonal,
    /// - S · Sᵀ represents the outer product of the softmax vector with itself.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the activations output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the activations input.
    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64> {
        let output = self
            .output
            .as_ref()
            .expect("output was not set. Please run the forward pass first.");
        let mut gradient = Array2::<f64>::uninit(output.raw_dim());

        for (i, (sample_softmax_out, d_value_sample)) in
            output.outer_iter().zip(d_values.outer_iter()).enumerate()
        {
            // What this does is compute the Jacobian matrix of the softmax function
            // and applying the chain rule.
            // The Jacobian matrix is computed as:
            // J = diag(S) - S · Sᵀ         where S is the softmax output.
            // The gradient is then computed as:
            // d_inputs = J · d_values
            // which is equivalent to:
            // d_inputs = (diag(S) - S · Sᵀ) · d_values
            // which simplifies to:
            // d_inputs = S ⊙ d_values - S * (S ⋅ d_values)
            // We do not need to transpose the S as it is a vector.
            let dot = sample_softmax_out.dot(&d_value_sample);

            let dinputs = &sample_softmax_out * &d_value_sample - &sample_softmax_out * dot;
            gradient
                .row_mut(i)
                .assign(&dinputs.mapv(std::mem::MaybeUninit::new));
        }

        unsafe { gradient.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arr_eq_approx;
    use ndarray::Array1;

    #[test]
    fn test_softmax() {
        let array = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 4.0],];

        let expected_output = ndarray::array![
            [0.0900305731703804, 0.2447284710547976, 0.6652409557748218],
            [0.2119415576170854, 0.5761168847658291, 0.2119415576170854],
        ];

        let output = Softmax::default().forward(&array);
        let sum = output.sum_axis(ndarray::Axis(1));

        assert_arr_eq_approx!(output, expected_output);
        assert_arr_eq_approx!(sum, Array1::<f64>::ones(sum.len()));
    }

    #[test]
    fn test_softmax_backward() {
        let softmax_out = ndarray::array![[5.0, 2.0, 3.0], [4.0, 5.0, 1.0],];

        let dvalues = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];

        let expected = ndarray::array![[-85., -32., -45.], [-172., -210., -41.]];

        let softmax = Softmax {
            output: Some(softmax_out),
        };
        let d = softmax.backward(&dvalues);

        assert_arr_eq_approx!(d, expected);
    }
}
