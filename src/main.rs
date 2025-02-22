mod layer;
mod initializer;
mod data;

use crate::layer::Layer;
use ndarray::s;

fn main() {
    let (x, y) = data::create_spiral_dataset(100, 3);
    data::display_spiral_dataset(&x, &y, "spiral.png").unwrap();

    let layer = layer::Dense::new::<initializer::Xavier>(2, 3);
    let layer_out = layer.forward(&x);

    println!("{:?}", layer_out.slice(s![..3, ..]));
}
