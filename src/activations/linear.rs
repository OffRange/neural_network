use crate::activations::ActivationFn;
use ndarray::Array2;

#[derive(Default)]
pub struct Linear;

impl ActivationFn for Linear {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn backward(&self, d_values: &Array2<f64>) -> Array2<f64> {
        d_values.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_forward() {
        let mut linear = Linear::default();

        let input = array![[1., 2., 3.], [4., 5., 6.]];
        let output = linear.forward(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_linear_backward() {
        let linear = Linear::default();

        let d_values = array![[1., 2., 3.], [4., 5., 6.]];
        let d = linear.backward(&d_values);
        assert_eq!(d, d_values);
    }
}
