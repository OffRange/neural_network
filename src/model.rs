use crate::data::Dataset;
use crate::loss::Loss;
use crate::metric::{Metric, ModelMetrics};
use crate::optimizers::Optimizer;
use crate::{Module, State};
use ndarray::{Array, Dimension};
use std::marker::PhantomData;

pub trait LayerChain<O> {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &Array<f64, Self::Input>) -> Array<f64, Self::Output>;
    fn backward(&mut self, grad: &Array<f64, Self::Output>) -> Array<f64, Self::Input>;

    fn update(&mut self, optimizer: &mut O);

    fn compile<T, L>(self, loss: L, optimizer: O) -> CompiledModel<Self, T, L, O>
    where
        L: Loss<T, PredDim = Self::Output>,
        O: Optimizer,
        T: Clone,
        Self: Sized,
    {
        CompiledModel::new(self, loss, optimizer)
    }

    fn update_state(&mut self, state: State);
}

impl<M, O> LayerChain<O> for M
where
    M: Module,
    O: Optimizer,
{
    type Input = M::Input;
    type Output = M::Output;

    fn forward(&mut self, input: &Array<f64, Self::Input>) -> Array<f64, Self::Output> {
        <M as Module>::forward(self, input)
    }

    fn backward(&mut self, grad: &Array<f64, Self::Output>) -> Array<f64, Self::Input> {
        <M as Module>::backward(self, grad)
    }

    fn update(&mut self, optimizer: &mut O) {
        if let Some(trainable) = self.as_trainable_mut() {
            optimizer.update(trainable)
        }
    }

    fn update_state(&mut self, state: State) {
        <M as Module>::update_state(self, state);
    }
}

impl<Head, Tail, O> LayerChain<O> for (Head, Tail)
where
    Head: LayerChain<O>,
    Tail: LayerChain<O, Input = Head::Output>,
    O: Optimizer,
{
    type Input = Head::Input;
    type Output = Tail::Output;

    fn forward(&mut self, input: &Array<f64, Self::Input>) -> Array<f64, Self::Output> {
        let (head, tail) = self;
        let output = head.forward(input);
        tail.forward(&output)
    }

    fn backward(&mut self, grad: &Array<f64, Self::Output>) -> Array<f64, Self::Input> {
        let (head, tail) = self;
        let grad = tail.backward(grad);
        head.backward(&grad)
    }

    fn update(&mut self, optimizer: &mut O) {
        let (head, tail) = self;
        head.update(optimizer);
        tail.update(optimizer);
    }

    fn update_state(&mut self, state: State) {
        let (head, tail) = self;
        head.update_state(state);
        tail.update_state(state);
    }
}

#[macro_export]
macro_rules! sequential {
    ($module:expr $(,)?) => {
        $module
    };

    ($module:expr, $($rest:expr),+ $(,)?) => {
        ($module, $crate::sequential!($($rest),+))
    };
}

pub struct CompiledModel<C, T, L, O>
where
    C: LayerChain<O>,
{
    _marker: PhantomData<T>,
    chain: C,
    loss: L,
    optimizer: O,
}

type AnyMetric<T, L> =
    Box<dyn Metric<T, PredDim = <L as Loss<T>>::PredDim, TargetDim = <L as Loss<T>>::TargetDim>>;

impl<C, T, L, O> CompiledModel<C, T, L, O>
where
    C: LayerChain<O>,
    L: Loss<T, PredDim = C::Output>,
    T: Clone,
    O: Optimizer,
{
    pub fn new(chain: C, loss: L, optimizer: O) -> Self {
        Self {
            _marker: PhantomData,
            chain,
            loss,
            optimizer,
        }
    }

    pub fn forward(&mut self, input: &Array<f64, C::Input>) -> Array<f64, C::Output> {
        self.chain.forward(input)
    }

    pub fn backward(&mut self, grad: &Array<f64, C::Output>) -> Array<f64, C::Input> {
        self.chain.backward(grad)
    }

    pub fn fit<D>(
        &mut self,
        dataset: &D,
        epochs: usize,
        batch_size: usize,
        shuffle: bool,
        print_every: usize,
        metrics: &[AnyMetric<T, L>],
    ) where
        D: Dataset<InType = f64, InDim = C::Input, OutDim = L::TargetDim, OutType = T>,
    {
        self.chain.update_state(State::Learning);

        let mut model_metrics = ModelMetrics::new(
            epochs,
            dataset.len().div_ceil(batch_size),
            metrics.len() + 1,
        );

        for epoch in 1..=epochs {
            self.optimizer.pre_update();
            let batch_metrics_data = dataset.batch_iter(batch_size, shuffle).map(|(x, y)| {
                let y_pred = self.forward(&x);

                let grad = self.loss.backwards(&y_pred, &y);
                self.backward(&grad);

                self.chain.update(&mut self.optimizer);

                let loss = self.loss.evaluate(&y_pred, &y);
                let metric_values = metrics
                    .iter()
                    .map(move |metric| metric.evaluate(&y_pred, &y));

                (loss, metric_values)
            });

            model_metrics.update(batch_metrics_data);
            if epoch == 1 || epoch % print_every == 0 {
                println!("{}", model_metrics.displayable(epoch));
            }
        }
    }

    pub fn evaluate<D>(&mut self, dataset: &D) -> (Array<D::InType, C::Output>, f64)
    where
        D: Dataset<InType = f64, InDim = C::Input, OutDim = L::TargetDim, OutType = T>,
    {
        self.chain.update_state(State::Evaluating);

        let pred = self.forward(&dataset.inputs().to_owned());
        let test_loss = self.loss.calculate(&pred, &dataset.outputs().to_owned());
        (pred, test_loss)
    }
}
