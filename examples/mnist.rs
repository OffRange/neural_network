use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array1, Array2, Axis, Ix1, Ix2, s};
use neural_network::Module;
use neural_network::data::{Dataset, NNDataset};
use neural_network::loss::Loss;
use neural_network::metric::Metric;
use neural_network::module::{activations, layers};
use neural_network::optimizers::Optimizer;
use neural_network::utils::Argmax;
use neural_network::{State, initializer, loss, metric, optimizers, regularizer};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read};

fn main() {
    let mnist = Mnist::new();

    let mut layer1 = layers::Dense::new_with_regularizers::<initializer::He>(
        784,
        1024,
        Some(Box::new(regularizer::L2::default())),
        Some(Box::new(regularizer::L2::default())),
    );
    let mut activation1 = activations::ReLU::default();
    let mut dropout1 = layers::Dropout::new(0.1);

    let mut layer2 = layers::Dense::new_with_regularizers::<initializer::He>(
        1024,
        512,
        Some(Box::new(regularizer::L2::default())),
        Some(Box::new(regularizer::L2::default())),
    );
    let mut activation2 = activations::ReLU::default();
    let mut dropout2 = layers::Dropout::new(0.1);

    let mut layer3 = layers::Dense::new_with_regularizers::<initializer::He>(
        512,
        256,
        Some(Box::new(regularizer::L2::default())),
        Some(Box::new(regularizer::L2::default())),
    );
    let mut activation3 = activations::ReLU::default();
    let mut dropout3 = layers::Dropout::new(0.1);

    let mut layer4 = layers::Dense::new_with_regularizers::<initializer::Xavier>(
        256,
        10,
        Some(Box::new(regularizer::L2::default())),
        Some(Box::new(regularizer::L2::default())),
    );
    let mut activation4 = activations::Softmax::default();

    let loss = loss::CategoricalCrossEntropy::new(1e-7);

    let mut optimizer = optimizers::Adam::new(0.0005, 1e-5, 1e-7, 0.9, 0.999);

    macro_rules! forward {
        (&$x:ident) => {{
            let layer1_out = layer1.forward(&$x);
            let activation1_out = activation1.forward(&layer1_out);
            let dropout1_out = dropout1.forward(&activation1_out);

            let layer2_out = layer2.forward(&dropout1_out);
            let activation2_out = activation2.forward(&layer2_out);
            let dropout2_out = dropout2.forward(&activation2_out);

            let layer3_out = layer3.forward(&dropout2_out);
            let activation3_out = activation3.forward(&layer3_out);
            let dropout3_out = dropout3.forward(&activation3_out);

            let layer4_out = layer4.forward(&dropout3_out);
            activation4.forward(&layer4_out)
        }};
    }

    let epochs = 300;
    for epoch in 1..=epochs {
        let mut loss_value = 0.;
        let mut acc_metric = 0.;
        let mut c = 0.;
        optimizer.pre_update();
        for (train_x, train_y) in mnist.train_dataset.batch_iter(64, true) {
            let out = forward!(&train_x);

            // Loss
            loss_value += loss.calculate(&out, &train_y);
            acc_metric += metric::MultiClassAccuracy.evaluate(&out, &train_y);
            c += 1.;

            // Backward
            let loss_back = loss.backwards(&out, &train_y);

            let activation4_back = activation4.backward(&loss_back);
            let layer4_back = layer4.backward(&activation4_back);

            let dropout3_back = dropout3.backward(&layer4_back);
            let activation3_back = activation3.backward(&dropout3_back);
            let layer3_back = layer3.backward(&activation3_back);

            let dropout2_back = dropout2.backward(&layer3_back);
            let activation2_back = activation2.backward(&dropout2_back);
            let layer2_back = layer2.backward(&activation2_back);

            let dropout1_back = dropout1.backward(&layer2_back);
            let activation1_back = activation1.backward(&dropout1_back);
            let _layer1_back = layer1.backward(&activation1_back);

            optimizer.update(&mut layer1);
            optimizer.update(&mut layer2);
            optimizer.update(&mut layer3);
            optimizer.update(&mut layer4);
        }

        if epochs < 100 || epoch % 100 == 0 || epoch == 1 {
            println!(
                "Epoch: {}/{}: AVG Loss: {:?}, AVG Accuracy: {:?}, lr: {}",
                epoch,
                epochs,
                loss_value / c,
                acc_metric / c,
                optimizer.learning_rate()
            );
        }
    }

    dropout1.update_state(State::Evaluating);
    dropout2.update_state(State::Evaluating);
    dropout3.update_state(State::Evaluating);

    let test_x = mnist.test_dataset.inputs().to_owned();
    let test_y = mnist.test_dataset.outputs().to_owned();
    let activation3_out = forward!(&test_x);

    let test_loss = loss.calculate(&activation3_out, &test_y);
    let acc_test_metric = metric::MultiClassAccuracy.evaluate(&activation3_out, &test_y);
    println!(
        "Test Loss: {:?}, Accuracy: {:?}",
        test_loss, acc_test_metric
    );

    println!("Real labels: {:?}", test_y.slice(s![0..10]));
    println!(
        "Prediction : {:?}",
        activation3_out.slice(s![0..10, ..]).argmax(Axis(1))
    );
}

type MnistDataset = NNDataset<f64, usize, Ix2, Ix1>;

/// MNIST dataset loader.
pub struct Mnist {
    pub train_dataset: MnistDataset,
    pub test_dataset: MnistDataset,
}

impl Default for Mnist {
    fn default() -> Self {
        let train_img = Mnist::load_img("examples/data/mnist/train-images.idx3-ubyte").unwrap();
        let train_lbl = Mnist::load_label("examples/data/mnist/train-labels.idx1-ubyte").unwrap();

        let test_img = Mnist::load_img("examples/data/mnist/t10k-images.idx3-ubyte").unwrap();
        let test_lbl = Mnist::load_label("examples/data/mnist/t10k-labels.idx1-ubyte").unwrap();

        Self {
            train_dataset: NNDataset::new(train_img, train_lbl),
            test_dataset: NNDataset::new(test_img, test_lbl),
        }
    }
}
impl Mnist {
    pub fn new() -> Self {
        Self::default()
    }

    fn load_img(path: &str) -> io::Result<Array2<f64>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header values
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2051 {
            panic!("Invalid magic number for image file: {}", magic);
        }

        let num_images = reader.read_u32::<BigEndian>()?;
        let rows = reader.read_u32::<BigEndian>()?;
        let cols = reader.read_u32::<BigEndian>()?;

        // Read all image data (each image is rows * cols bytes)
        let mut images = vec![0u8; (num_images * rows * cols) as usize];
        reader.read_exact(&mut images)?;

        Ok(
            Array2::from_shape_vec((num_images as usize, (rows * cols) as usize), images)
                .unwrap()
                .mapv(|x| x as f64 / 255.0),
        )
    }

    fn load_label(path: &str) -> io::Result<Array1<usize>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header values
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2049 {
            panic!("Invalid magic number for label file: {}", magic);
        }
        let num_labels = reader.read_u32::<BigEndian>()?;

        // Read all label data (each label is one byte)
        let mut labels = vec![0u8; num_labels as usize];
        reader.read_exact(&mut labels)?;

        Ok(labels.iter().map(|&x| x as usize).collect())
    }
}
