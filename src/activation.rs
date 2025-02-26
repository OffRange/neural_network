use crate::expect;
use ndarray::{Array2, ArrayView2};

pub trait ActivationFn<'a>: Default {
    /// Performs the forward pass for the activation function.
    ///
    /// # Arguments
    ///
    /// * `input` - An `ArrayView2<f64>` representing the input data.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the activated output.
    fn forward(&mut self, input: ArrayView2<'a, f64>) -> Array2<f64>;

    /// Performs the backward pass for the activation function.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the activation output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the activation input.
    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64>;
}

#[derive(Default)]
pub struct ReLU<'a> {
    input: Option<ArrayView2<'a, f64>>,
}

impl<'a> ActivationFn<'a> for ReLU<'a> {
    fn forward(&mut self, input: ArrayView2<'a, f64>) -> Array2<f64> {
        self.input = Some(input);
        input.map(|x| x.max(0.0))
    }

    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64> {
        expect!(self.input).mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * d_values
    }
}

pub struct LeakyReLU<'a> {
    alpha: f64,
    input: Option<ArrayView2<'a, f64>>,
}

impl Default for LeakyReLU<'_> {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl LeakyReLU<'_> {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            input: None,
        }
    }
}

impl<'a> ActivationFn<'a> for LeakyReLU<'a> {
    fn forward(&mut self, input: ArrayView2<'a, f64>) -> Array2<f64> {
        self.input = Some(input);
        input.map(|&x| if x >= 0.0 { x } else { x * self.alpha })
    }

    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64> {
        expect!(self.input).mapv(|x| if x >= 0.0 { 1.0 } else { self.alpha }) * d_values
    }
}

#[derive(Default)]
pub struct Softmax {
    output: Option<Array2<f64>>,
}

impl ActivationFn<'_> for Softmax {
    fn forward(&mut self, input: ArrayView2<'_, f64>) -> Array2<f64> {
        let max = input.map_axis(ndarray::Axis(1), |row| {
            row.iter().copied().reduce(f64::max).unwrap()
        }).insert_axis(ndarray::Axis(1));


        let exp = (&input - max).exp();
        let sum = exp.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        let out = exp / sum;
        self.output = Some(out.clone());
        out
    }


    /// Performs the backward pass for the activation function.
    ///
    /// For a softmax output S, where `S[i][j]` is the probability of the j-th class for the i-th sample,
    /// the derivative with respect to the input z is given by:
    /// $$
    ///
    ///     ∂S[i][j] / ∂z[i][k] = S[i][j] * (δ[j][k] - S[i][k])
    ///
    /// $$
    ///
    /// Here, `δ[j][k]` is the Kronecker delta, which equals 1 if j = k and 0 otherwise.
    ///
    /// In matrix form, the Jacobian can be expressed as:
    ///
    ///     J = diag(S) - S · Sᵀ
    ///
    /// where:
    /// - diag(S) is a diagonal matrix with the softmax outputs along the diagonal,
    /// - S · Sᵀ represents the outer product of the softmax vector with itself.
    ///
    /// # Arguments
    ///
    /// * `d_values` - A reference to an `Array2<f64>` representing the gradient of the loss with respect to the activation output.
    ///
    /// # Returns
    ///
    /// * An `Array2<f64>` representing the gradient of the loss with respect to the activation input.
    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64> {
        let output = expect!(self.output.as_ref());
        let mut gradient = Array2::<f64>::uninit(output.raw_dim());

        for (i, (sample_softmax_out, d_value_sample)) in output.outer_iter().zip(d_values.outer_iter()).enumerate() {
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
            gradient.row_mut(i).assign(&dinputs.mapv(|x| std::mem::MaybeUninit::new(x)));
        }

        unsafe { gradient.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_arr_eq_approx, assert_eq_approx};
    use ndarray::Array1;

    #[test]
    fn test_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, 0.0], [0.0, 4.0]];
        let output = ReLU::default().forward(input.view());
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_relu_backward() {
        let input = ndarray::array![
            [1., 2., -3., -4.],
            [2., -7., -1., 3.],
            [-1., 2., 5., -1.]
        ];

        let dvalues = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.]
        ];

        let expected = ndarray::array![
            [1., 2., 0., 0.],
            [5., 0., 0., 8.],
            [0., 10., 11., 0.]
        ];

        let mut relu = ReLU::default();
        let _ = relu.forward(input.view());
        let d = relu.backward(&dvalues);

        assert_eq!(d, expected);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, -0.02], [-0.03, 4.0]];
        let output = LeakyReLU::new(0.01).forward(input.view());
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_leaky_relu_backwards() {
        let input = ndarray::array![
            [1., 2., -3., -4.],
            [2., -7., -1., 3.],
            [-1., 2., 5., -1.]
        ];

        let dvalues = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.]
        ];

        let expected = ndarray::array![
            [1., 2., 0.03, 0.04],
            [5., 0.06, 0.07, 8.],
            [0.09, 10., 11., 0.12]
        ];

        let mut leaky_relu = LeakyReLU::new(0.01);
        let _ = leaky_relu.forward(input.view());
        let d = leaky_relu.backward(&dvalues);

        assert_eq!(d, expected);
    }

    #[test]
    fn test_softmax() {
        let array = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 4.0],
        ];

        let expected_output = ndarray::array![
            [0.0900305731703804, 0.2447284710547976, 0.6652409557748218],
            [0.2119415576170854, 0.5761168847658291, 0.2119415576170854],
        ];

        let output = Softmax::default().forward(array.view());
        let sum = output.sum_axis(ndarray::Axis(1));

        assert_arr_eq_approx!(output, expected_output);
        assert_arr_eq_approx!(sum, Array1::<f64>::ones(sum.len()));
    }

    #[test]
    fn test_softmax_backward() {
        let softmax_out = ndarray::array![
            [5.0, 2.0, 3.0],
            [4.0, 5.0, 1.0],
        ];

        let dvalues = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];

        let expected = ndarray::array![
            [-85.,  -32.,  -45.],
            [-172., -210.,  -41.]
        ];

        let mut softmax = Softmax::default();
        softmax.output = Some(softmax_out);
        let d = softmax.backward(&dvalues);

        assert_arr_eq_approx!(d, expected);
    }
}