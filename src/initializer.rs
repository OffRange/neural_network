use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};

pub trait Initializer {
    fn new(fan_in: usize, fan_out: usize) -> Self;
    fn dist(&self) -> impl Distribution<f64>;
}

pub struct Xavier {
    uniform: Uniform<f64>,
}

impl Initializer for Xavier {
    fn new(fan_in: usize, fan_out: usize) -> Self {
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
        Self {
            uniform: Uniform::new(-bound, bound),
        }
    }

    fn dist(&self) -> impl Distribution<f64> {
        self.uniform
    }
}

pub struct He {
    normal: Normal<f64>,
}

impl Initializer for He {
    fn new(fan_in: usize, _fan_out: usize) -> Self {
        let std_dev = (2.0 / fan_in as f64).sqrt();

        Self {
            normal: Normal::new(0.0, std_dev).unwrap(),
        }
    }

    fn dist(&self) -> impl Distribution<f64> {
        self.normal
    }
}


#[cfg(test)]
pub(crate) mod test {
    use crate::initializer::Initializer;
    use ndarray_rand::rand::prelude::*;

    struct ConstantDistribution {
        constant: f64,
    }

    impl Distribution<f64> for ConstantDistribution {
        fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
            self.constant
        }
    }

    pub(crate) struct ConstantInitializer<const C: usize> {
        constant: usize,
    }

    impl<const C: usize> Initializer for ConstantInitializer<C> {
        fn new(_fan_in: usize, _fan_out: usize) -> Self {
            Self { constant: C }
        }

        fn dist(&self) -> impl Distribution<f64> {
            ConstantDistribution {
                constant: self.constant as f64,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand::prelude::*;

    #[test]
    fn test_xavier_initializer_bounds() {
        let fan_in = 4;
        let fan_out = 6;
        let xavier = Xavier::new(fan_in, fan_out);
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let dist = xavier.dist();

        // Use a seeded RNG for reproducibility.
        let mut rng = StdRng::seed_from_u64(42);


        // Sample multiple values and ensure they are within the expected range.
        for _ in 0..1_000_000 {
            let sample = dist.sample(&mut rng);
            assert!(
                sample >= -bound && sample < bound,
                "Xavier sample {} not in range [-{}, {})",
                sample,
                bound,
                bound
            );
        }
    }

    #[test]
    fn test_he_initializer_statistics() {
        let fan_in = 4;
        let he = He::new(fan_in, 0);
        let expected_std = (2.0 / fan_in as f64).sqrt();
        let dist = he.dist();

        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<f64> = (0..1_000_000).map(|_| dist.sample(&mut rng)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let empirical_std = variance.sqrt();

        // Check that the empirical mean is close to 0.
        assert!(
            mean.abs() < 0.1,
            "Empirical mean {} is not close to 0",
            mean
        );

        // Check that the empirical standard deviation is close to the expected value.
        assert!(
            (empirical_std - expected_std).abs() < 0.1,
            "Empirical std {} not close to expected std {}",
            empirical_std,
            expected_std
        );
    }
}