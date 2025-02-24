use ndarray::{Array1, Array2};

pub fn argmax(x: &Array2<f64>) -> Array1<usize> {
    x.map_axis(ndarray::Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    })
}

#[cfg(test)]
mod tests {
    use ndarray::array;

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
}