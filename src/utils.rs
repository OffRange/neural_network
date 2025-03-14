use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dimension, Ix};

pub trait Argmax<D>
where
    D: Dimension,
{
    fn argmax(&self, axis: Axis) -> Array<usize, D::Smaller>;
}

impl<S, D> Argmax<D> for ArrayBase<S, D>
where
    S: ndarray::Data<Elem = f64>,
    D: Dimension + ndarray::RemoveAxis,
{
    fn argmax(&self, axis: Axis) -> Array<usize, D::Smaller> {
        self.map_axis(axis, |axis| {
            axis.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
    }
}

pub trait OneHotNumber {}

impl OneHotNumber for Ix {}
impl OneHotNumber for f64 {}

pub trait ToOneHot<N>
where
    N: OneHotNumber,
{
    fn to_one_hot(&self, n_classes: usize) -> Array2<N>;
}

impl ToOneHot<f64> for Array1<Ix> {
    fn to_one_hot(&self, n_classes: usize) -> Array2<f64> {
        let mut one_hot = Array2::zeros((self.len(), n_classes));

        for (i, &index) in self.iter().enumerate() {
            one_hot[[i, index]] = 1.0;
        }

        one_hot
    }
}

impl ToOneHot<Ix> for Array1<Ix> {
    fn to_one_hot(&self, n_classes: usize) -> Array2<Ix> {
        let mut one_hot = Array2::zeros((self.len(), n_classes));

        for (i, &index) in self.iter().enumerate() {
            one_hot[[i, index]] = 1;
        }

        one_hot
    }
}

#[macro_export]
macro_rules! expect {
    ($t:expr) => {
        $t.expect(concat!(
            stringify!($t),
            " was not set. Please run the forward pass first."
        ))
    };
}

#[cfg(test)]
mod tests {
    use super::{Argmax, ToOneHot};
    use ndarray::{Array2, Axis, Ix, array};

    #[test]
    fn test_argmax() {
        let arr = array![
            [[0.7], [0.2], [0.1]],
            [[0.5], [0.1], [0.4]],
            [[0.02], [0.9], [0.08]]
        ];

        let result_ax0 = arr.argmax(Axis(0));
        let result_ax1 = arr.argmax(Axis(1));
        let result_ax2 = arr.argmax(Axis(2));

        assert_eq!(result_ax0, array![[0], [2], [1]]);
        assert_eq!(result_ax1, array![[0], [0], [1]]);
        assert_eq!(result_ax2, Array2::<usize>::zeros((3, 3)));
    }

    #[test]
    fn test_to_one_hot() {
        let array = array![0, 1, 2, 0, 1, 2];
        let one_hot: Array2<Ix> = array.to_one_hot(3);
        let one_hot_f64: Array2<f64> = array.to_one_hot(3);

        let expected = array![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ];

        assert_eq!(one_hot, expected);
        assert_eq!(one_hot_f64, expected.mapv(|x| x as f64));
    }
}
