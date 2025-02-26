mod layer;
mod initializer;
mod data;
mod activation;
mod assert;
mod loss;
mod metric;
mod utils;

use crate::activation::ActivationFn;
use crate::layer::Layer;
use crate::loss::Loss;
use crate::metric::Metric;
use std::ops::Rem;

fn main() {
    let (x, y) = data::create_spiral_dataset(100, 3);
    data::display_spiral_dataset(&x, &y, "spiral.png").unwrap();

    let mut layer1 = layer::Dense::new::<initializer::Xavier>(2, 3);
    let mut activation1 = activation::ReLU::default();

    let mut layer2 = layer::Dense::new::<initializer::Xavier>(3, 3);
    let mut activation2 = activation::Softmax::default();

    let loss = loss::CategoricalCrossEntropy::default();

    // Forward
    let layer1_output = layer1.forward(x.view());
    let activation1_output = activation1.forward(layer1_output.view());

    let layer2_output = layer2.forward(activation1_output.view());
    let activation2_output = activation2.forward(layer2_output.view());

    let loss_value = loss.calculate(&activation2_output, &y);
    let acc_metric = metric::MultiClassAccuracy::default().evaluate(&activation2_output, &y);

    // Backward
    let loss_back = loss.backwards(&activation2_output, &y);

    let activation2_back = activation2.backward(&loss_back);
    let layer2_back = layer2.backward(&activation2_back, 0.01);

    let activation1_back = activation1.backward(&layer2_back);
    let _layer1_back = layer1.backward(&activation1_back, 0.01);

    println!("Loss: {:.4}, Accuracy: {:.4}", loss_value, acc_metric);
}
