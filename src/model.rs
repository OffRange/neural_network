use crate::Module;
use crate::data::Dataset;
use crate::loss::Loss;
use crate::optimizers::Optimizer;
use ndarray::{Array2, Dimension, Ix2};
use std::marker::PhantomData;

pub struct ModelArchitecture<O, LD, L>
where
    O: Optimizer,
    L: Loss<LD, f64>,
    LD: Dimension,
{
    _marker: PhantomData<LD>,
    modules: Vec<Box<dyn Module>>,
    optimizer: O,
    loss: L,
}

impl<O, LD, L> ModelArchitecture<O, LD, L>
where
    O: Optimizer,
    L: Loss<LD, f64>,
    LD: Dimension,
{
    pub fn new(optimizer: O, loss: L) -> Self {
        Self {
            _marker: PhantomData,
            modules: Vec::new(),
            optimizer,
            loss,
        }
    }

    pub fn add_module<M>(&mut self, module: M)
    where
        M: Module + 'static,
    {
        self.modules.push(Box::new(module));
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut output = x.to_owned();
        for module in self.modules.iter_mut() {
            output = module.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, grad: &Array2<f64>) {
        let mut grad = grad.to_owned();
        for module in self.modules.iter_mut().rev() {
            grad = module.backward(&grad);
        }
    }

    pub fn fit<D>(
        &mut self,
        dataset: &D,
        epochs: usize,
        batch_size: usize,
        shuffle: bool,
        print_every: usize,
    ) where
        D: Dataset<InType = f64, InDim = Ix2, OutDim = LD, OutType = f64>,
    {
        for epoch in 0..epochs {
            let mut loss = 0.0;
            for (x, y) in dataset.batch_iter(batch_size, shuffle) {
                let y_pred = self.forward(&x);

                loss += self.loss.calculate(&y_pred, &y);

                let grad = self.loss.backwards(&y_pred, &y);
                self.backward(&grad);

                for module in self.modules.iter_mut() {
                    if let Some(trainable) = module.as_trainable_mut() {
                        self.optimizer.update(trainable);
                    }
                }
            }

            if epoch % print_every == 0 {
                println!("Epoch: {}, Loss: {}", epoch, loss);
            }
        }
    }
}
