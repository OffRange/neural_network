use crate::{Module, State};
use ndarray::Array2;

#[derive(Default)]
pub struct Sigmoid {
    output: Option<Array2<f64>>,
}

impl Module for Sigmoid {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, d_values: &Array2<f64>) -> Array2<f64> {
        let output = self
            .output
            .as_ref()
            .expect("output was not set. Please run the forward pass first.");
        let gradient = output.mapv(|x| x * (1.0 - x));
        d_values * gradient
    }

    fn update_state(&mut self, _state: State) {}
}

#[cfg(test)]
mod tests {
    use super::Sigmoid;
    use crate::{Module, assert_arr_eq_approx};
    use ndarray::array;

    #[test]
    fn test_sigmoid_forward() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let expected = array![
            [0.7310585786300049, 0.8807970779778823],
            [0.9525741268224334, 0.9820137900379085]
        ];

        let output = Sigmoid::default().forward(&input);

        assert_arr_eq_approx!(expected, output);
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut sigmoid = Sigmoid::default();
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let _ = sigmoid.forward(&input);

        let d_values = array![[0.25, 0.5], [0.75, 1.]];
        let expected = array![
            [0.04915298331037046, 0.05249679270175331],
            [0.033882494798184004, 0.017662706213291107]
        ];

        let output = sigmoid.backward(&d_values);

        assert_arr_eq_approx!(expected, output);
    }
}
