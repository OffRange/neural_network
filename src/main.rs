#[cfg(feature = "blas")]
extern crate blas_src;

mod layer;
mod initializer;
mod data;
mod activation;
mod assert;
mod loss;
mod metric;
mod utils;
mod optimizer;
mod regularizer;

use crate::activation::ActivationFn;
use crate::layer::Layer;
use crate::loss::Loss;
use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::utils::argmax;
use ndarray::{array, Array1, Array2};
use ndarray_npy::write_npy;
use rand::seq::SliceRandom;
use std::fs;

fn shuffle_rows(x: &mut Array2<f64>, y: &mut Array1<usize>) {
    assert_eq!(x.nrows(), y.len(), "The number of rows in the input array must match the number of labels.");

    let nrows = x.nrows();
    let mut indices: Vec<usize> = (0..nrows).collect();
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);

    // Create a new array with the same shape.
    let mut shuffled_x = Array2::<f64>::default(x.raw_dim());
    let mut shuffled_y = Array1::<usize>::default(y.raw_dim());

    // For each new row, assign from the shuffled index.
    for (i, &row_idx) in indices.iter().enumerate() {
        // Copy the row from the original array.
        let row = x.row(row_idx);
        shuffled_x.row_mut(i).assign(&row);
        shuffled_y[i] = y[row_idx];
    }

    // Overwrite the original array with the shuffled array.
    *x = shuffled_x;
    *y = shuffled_y;
}

#[macro_export]
macro_rules! write_npy {
    (forward: $epoch:ident, $layer:ident) => {
        $crate::write_npy!($layer, "forward", $epoch);
    };

    (backward: $epoch:ident, $layer:ident) => {
        $crate::write_npy!($layer, "backward", $epoch);
    };

    ($layer:ident, $t:literal, $epoch:ident) => {
        $crate::write_npy!{
            $layer, $t, $epoch,

            fn weights
            fn biases
            fn weights_gradient
            fn biases_gradient
            fn weight_momentum
            fn bias_momentum
            fn weight_cache
            fn bias_cache
        }
    };

    ($layer:ident, $t:literal, $epoch:ident, $(fn $name:ident)+) => {

        #[cfg(feature = "print-debug")]
        {
            println!("Writing {} for epoch {} - {}", stringify!($layer), $epoch, $t);
        $(
            write_npy(format!("layers/{}_{}_{}_{}.npy", stringify!($layer), $t, stringify!($name), $epoch), $layer.$name()).unwrap();
        )+
        }
    };
}

fn main() {
    #[cfg(feature = "print-debug")]
    println!("Feature enabled: print-debug");
    #[cfg(feature = "blas")]
    println!("Feature enabled: blas");
    #[cfg(feature = "blas-accelerate")]
    println!("Feature enabled: blas-accelerate");

    fs::remove_dir_all("layers").unwrap();
    fs::create_dir("layers").unwrap();
    println!("Creating spiral dataset...");
    let (mut x, mut y) = data::create_spiral_dataset(1000, 3);

    println!("Shuffling dataset...");
    shuffle_rows(&mut x, &mut y);

    println!("Saving to npy");
    write_npy("data_x.npy", &x).unwrap();
    write_npy("data_y.npy", &y.mapv(|x| x as u32)).unwrap();

    println!("Saving dataset to spiral.png...");
    data::display_spiral_dataset(&x, &y, "spiral.png").unwrap();

    let mut layer1 = layer::Dense::new_with_regularizers::<initializer::He>(2, 512, Some(Box::new(regularizer::L2::new(5e-4))), None);
    let mut activation1 = activation::ReLU::default();

    let mut layer2 = layer::Dense::new_with_regularizers::<initializer::Xavier>(512, 3, Some(Box::new(regularizer::L2::new(5e-4))), None);
    let mut activation2 = activation::Softmax::default();

    let loss = loss::CategoricalCrossEntropy::new(1e-7);

    let mut optimizer = optimizer::Adam::new(0.02, 1e-5, 1e-7, 0.9, 0.999);
    //let mut optimizer = optimizer::SGD::new(1., 0., 1e-5);

    {
        write_npy("weights1.npy", layer1.weights()).unwrap();
        write_npy("biases1.npy", layer1.biases()).unwrap();

        write_npy("weights2.npy", layer2.weights()).unwrap();
        write_npy("biases2.npy", layer2.biases()).unwrap();
    }

    let epochs = 10_000;
    for epoch in 1..=epochs {
        // Forward
        let layer1_output = layer1.forward(&x);
        let activation1_output = activation1.forward(&layer1_output);

        let layer2_output = layer2.forward(&activation1_output);
        let activation2_output = activation2.forward(&layer2_output);

        let loss_value = loss.calculate(&activation2_output, &y);
        let acc_metric = metric::MultiClassAccuracy::default().evaluate(&activation2_output, &y);

        if epoch % 100 == 0 || epoch == 1 || cfg!(feature = "print-debug") {
            println!("Epoch: {}/{}: Loss: {:?}, Accuracy: {:?}, lr: {}", epoch, epochs, loss_value, acc_metric, optimizer.learning_rate());
            write_npy!(forward: epoch, layer1);
            write_npy!(forward: epoch, layer2);
        }

        // Backward
        let loss_back = loss.backwards(&activation2_output, &y);

        let activation2_back = activation2.backward(&loss_back);
        let layer2_back = layer2.backward(&activation2_back);

        let activation1_back = activation1.backward(&layer2_back);
        let _layer1_back = layer1.backward(&activation1_back);

        optimizer.pre_update();
        optimizer.update(&mut layer1);
        optimizer.update(&mut layer2);

        write_npy!(backward: epoch, layer1);
        write_npy!(backward: epoch, layer2);
    }

    // Test dataset
    let (x, y) = data::create_spiral_dataset(100, 3);
    let mut forward = |x: &Array2<f64>, y| {
        let layer1_output = layer1.forward(&x);
        let activation1_output = activation1.forward(&layer1_output);
        let layer2_output = layer2.forward(&activation1_output);
        let activation2_output = activation2.forward(&layer2_output);

        let loss_value = loss.calculate(&activation2_output, y);
        let acc_metric = metric::MultiClassAccuracy::default().evaluate(&activation2_output, y);

        (activation2_output, loss_value, acc_metric)
    };
    let (_, loss_value, acc_metric) = forward(&x, &y);
    println!("Test Loss: {:?}, Accuracy: {:?}", loss_value, acc_metric);

    data::visualize_pred(&x, &y, |x, y| {
        let input = array![[x, y]];
        let layer1_output = layer1.forward(&input);
        let activation1_output = activation1.forward(&layer1_output);
        let layer2_output = layer2.forward(&activation1_output);
        let activation2_output = activation2.forward(&layer2_output);

        let pred = argmax(&activation2_output);
        pred[0]
    }).unwrap();
}