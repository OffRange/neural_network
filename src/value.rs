use ndarray::{ArrayBase, Data, Dimension};
use std::fmt::Display;

pub trait Value: Display {
    fn value(&self) -> f64;
    fn name(&self) -> &'static str;
}

pub trait ValueMean {
    fn value_mean(&self) -> f64;
}

impl<V, S, D> ValueMean for ArrayBase<S, D>
where
    S: Data<Elem = V>,
    D: Dimension,
    V: Value,
{
    fn value_mean(&self) -> f64 {
        self.iter().map(|v| v.value()).sum::<f64>() / self.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use crate::value::{Value, ValueMean};
    use ndarray::array;
    use std::fmt::{Display, Formatter};

    struct TestValue(f64);

    impl TestValue {
        fn new(value: f64) -> Self {
            Self(value)
        }
    }

    impl Display for TestValue {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            unreachable!()
        }
    }

    impl Value for TestValue {
        fn value(&self) -> f64 {
            self.0
        }

        fn name(&self) -> &'static str {
            "TestValue"
        }
    }

    #[test]
    fn test_mean() {
        let raw_values = array![1., 2., 3., 4., 5.];
        let values = array![
            TestValue::new(1.),
            TestValue::new(2.),
            TestValue::new(3.),
            TestValue::new(4.),
            TestValue::new(5.)
        ];

        assert_eq!(values.value_mean(), raw_values.mean().unwrap());
    }
}
