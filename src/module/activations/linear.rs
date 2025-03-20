use crate::{Module, State};
use ndarray::{Array, Dimension};
use std::marker::PhantomData;

#[derive(Default)]
pub struct Linear<D> {
    _marker: PhantomData<D>,
}

impl<D> Module for Linear<D>
where
    D: Dimension,
{
    type Input = D;
    type Output = D;

    fn forward(&mut self, x: &Array<f64, D>) -> Array<f64, D> {
        x.clone()
    }

    fn backward(&mut self, d_values: &Array<f64, D>) -> Array<f64, D> {
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
        let output = Linear::default().forward(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_linear_backward() {
        let d_values = array![[1., 2., 3.], [4., 5., 6.]];
        let d = Linear::default().backward(&d_values);
        assert_eq!(d, d_values);
    }
}
