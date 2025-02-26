use ndarray::Array2;

pub trait ActivationFn: Default {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64>;

    /// Computes the gradient of the activation function with respect to the input,
    ///
    /// # Arguments
    /// `input` - The input to the activation function. This is typically the output
    /// of the previous layer.
    ///
    /// `value` - The gradient of the loss function with respect to the output of this layer.
    fn backward(&self, input: &Array2<f64>, value: &Array2<f64>) -> Array2<f64>;
}

#[derive(Default)]
pub struct ReLU;

impl ActivationFn for ReLU {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.map(|x| x.max(0.0))
    }

    fn backward(&self, input: &Array2<f64>, value: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * value
    }
}

pub struct LeakyReLU {
    alpha: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
        }
    }
}

impl ActivationFn for LeakyReLU {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.map(|&x| if x >= 0.0 { x } else { x * self.alpha })
    }

    fn backward(&self, input: &Array2<f64>, value: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x >= 0.0 { 1.0 } else { self.alpha }) * value
    }
}

#[derive(Default)]
pub struct Softmax;

impl ActivationFn for Softmax {
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let max = input.map_axis(ndarray::Axis(1), |row| {
            row.iter().copied().reduce(f64::max).unwrap()
        }).insert_axis(ndarray::Axis(1));

        let exp = (input - max).exp();
        let sum = exp.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        exp / sum
    }

    /// Computes the gradient of the activation function with respect to the input,
    ///
    /// For a softmax output S, where `S[i][j]` is the probability of the j-th class for the i-th sample,
    /// the derivative with respect to the input z is given by:
    ///
    ///     ∂S[i][j] / ∂z[i][k] = S[i][j] * (δ[j][k] - S[i][k])
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
    /// `input` - The input to the activation function. This is typically the output
    /// of the previous layer.
    ///
    /// `value` - The gradient of the loss function with respect to the output of this layer.
    fn backward(&self, input: &Array2<f64>, value: &Array2<f64>) -> Array2<f64> {
        let mut gradient = Array2::<f64>::uninit(input.raw_dim());

        for (i, (sample_softmax_out, input_sample)) in input.outer_iter().zip(value.outer_iter()).enumerate() {
            // What this does is compute the Jacobian matrix of the softmax function
            // and applying the chain rule.
            // The Jacobian matrix is computed as:
            // J = diag(S) - S · Sᵀ         where S is the softmax output.
            // The gradient is then computed as:
            // d_inputs = J · d_values
            // which is equivalent to:
            // d_inputs = (diag(S) - S · Sᵀ) · d_values
            // which simplifies to:
            // d_inputs = S ⊙ input_sample - S * (S ⋅ input_sample)
            // We do not need to transpose the S as it is a vector.
            let dot = sample_softmax_out.dot(&input_sample);

            let dinputs = &sample_softmax_out * &input_sample - &sample_softmax_out * dot;
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
        let output = ReLU::default().forward(&input);
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

        let d = ReLU::default().backward(&input, &dvalues);

        assert_eq!(d, expected);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let input = ndarray::array![[1.0, -2.0], [-3.0, 4.0]];
        let expected_output = ndarray::array![[1.0, -0.02], [-0.03, 4.0]];
        let output = LeakyReLU::new(0.01).forward(&input);
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

        let d = LeakyReLU::new(0.01).backward(&input, &dvalues);

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

        let output = Softmax::default().forward(&array);
        let sum = output.sum_axis(ndarray::Axis(1));

        assert_arr_eq_approx!(output, expected_output);
        assert_arr_eq_approx!(sum, Array1::<f64>::ones(sum.len()));
    }

    #[test]
    fn test_softmax_backward() {
        let input = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 4.0],
        ];

        let dvalues = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];

        let expected = ndarray::array![
            [-13., -24., -33.],
            [-244., -300., -236.],
        ];

        let d = Softmax::default().backward(&input, &dvalues);

        assert_arr_eq_approx!(d, expected);
    }
}