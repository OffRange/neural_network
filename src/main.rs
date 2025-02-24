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
use ndarray::Axis;

fn main() {
    let (x, y) = data::create_spiral_dataset(100, 3);
    data::display_spiral_dataset(&x, &y, "spiral.png").unwrap();

    let layer = layer::Dense::new::<initializer::Xavier>(2, 3);
    let layer2 = layer::Dense::new::<initializer::Xavier>(3, 3);

    let layer_out = layer.forward(&x.select(Axis(0), &[50, 150, 250]));
    let layer_out_relu = activation::ReLU.forward(&layer_out);

    let layer2_out = layer2.forward(&layer_out_relu);
    let layer2_out_softmax = activation::Softmax.forward(&layer2_out);

    let loss = loss::CategoricalCrossEntropy::default();
    let loss = loss.calculate(&layer2_out_softmax, &y.select(Axis(0), &[50, 150, 250]));

    let acc = metric::MultiClassAccuracy::default();
    let acc = acc.evaluate(&layer2_out_softmax, &y.select(Axis(0), &[50, 150, 250]));

    println!("Accuracy [MultiClassAccuracy]: {:?}", acc);
    println!("Loss [CategoricalCrossEntropy]: {:?}", loss);
    println!("Layer2 out softmax:\n{:?}", layer2_out_softmax);
}
