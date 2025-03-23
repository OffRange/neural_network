use crate::metric::MetricValue;
use crate::value::{Value, ValueMean};
use ndarray::{Array, ArrayView, Ix2, Ix3, s};
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct ModelMetrics {
    epoch: usize,
    metrics: Array<MetricValue, Ix3>,
}

pub struct DisplayableMetrics<'a> {
    metrics: ArrayView<'a, MetricValue, Ix2>,
    epoch: usize,
    total_epochs: usize,
}

impl<'a> DisplayableMetrics<'a> {
    pub fn new(metrics: &'a Array<MetricValue, Ix3>, epoch: usize) -> Self {
        Self {
            metrics: metrics.slice(s![epoch - 1, .., ..]),
            epoch,
            total_epochs: metrics.shape()[0],
        }
    }
}

impl Display for DisplayableMetrics<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (_, metric_size) = self.metrics.dim();
        write!(f, "Epoch {}/{}", self.epoch, self.total_epochs)?;

        let loss = self.metrics.slice(s![.., 0]);
        write!(
            f,
            ", Loss [{}]: {:.8}",
            loss.first().expect("No loss found").name(),
            loss.value_mean()
        )?;

        for i in 1..metric_size {
            let metric_values = self.metrics.slice(s![.., i]);
            write!(
                f,
                ", {}: {:.8}",
                metric_values.first().expect("No metric at found").name(),
                metric_values.value_mean()
            )?;
        }
        Ok(())
    }
}

impl ModelMetrics {
    pub fn new(epochs: usize, batch_size: usize, metric_size: usize) -> Self {
        Self {
            epoch: 0,
            metrics: Array::default((epochs, batch_size, metric_size)),
        }
    }

    pub fn update<I, J>(&mut self, data: I)
    where
        I: IntoIterator<Item = (MetricValue, J)>,
        J: IntoIterator<Item = MetricValue>,
    {
        let current_epoch = self.epoch;
        for (i, (batch_loss, batch_metric_values)) in data.into_iter().enumerate() {
            let mut batch_data = self.metrics.slice_mut(s![current_epoch, i, ..]);

            if let Some(first) = batch_data.first_mut() {
                *first = batch_loss;
            }

            for (cell, value) in batch_data.iter_mut().skip(1).zip(batch_metric_values) {
                *cell = value;
            }
        }

        self.epoch += 1;
    }

    pub fn displayable(&self, epoch: usize) -> DisplayableMetrics {
        DisplayableMetrics::new(&self.metrics, epoch)
    }
}
