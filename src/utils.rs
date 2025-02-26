use ndarray::{Array1, Array2, Ix};

pub fn argmax(x: &Array2<f64>) -> Array1<usize> {
    x.map_axis(ndarray::Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    })
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
        $t.expect(concat!(stringify!($t), " was not set. Please run the forward pass first."))
    };
}

#[cfg(test)]
mod tests {
    use super::ToOneHot;
    use ndarray::{array, Array2, Ix};

    #[test]
    fn test_argmax() {
        let array = array![
            [0.1, 0.2, 0.3],
            [0.6, 0.5, 0.4],
            [0.7, 0.9, 0.8],
        ];

        let result = super::argmax(&array);

        assert_eq!(result, array![2, 0, 1]);
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