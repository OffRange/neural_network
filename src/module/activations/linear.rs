use crate::{Module, State};
use ndarray::Array2;

#[derive(Default)]
pub struct Linear;

impl Module for Linear {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn backward(&mut self, d_values: &Array2<f64>) -> Array2<f64> {
        d_values.clone()
    }

    fn update_state(&mut self, _state: State) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_forward() {
        let input = array![[1., 2., 3.], [4., 5., 6.]];
        let output = Linear.forward(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_linear_backward() {
        let d_values = array![[1., 2., 3.], [4., 5., 6.]];
        let d = Linear.backward(&d_values);
        assert_eq!(d, d_values);
    }
}
