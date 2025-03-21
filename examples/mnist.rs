use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{s, Array1, Array2, Axis, Ix1, Ix2};
use neural_network::data::{Dataset, NNDataset};
use neural_network::model::LayerChain;
use neural_network::module::{activations, layers};
use neural_network::sequential;
use neural_network::utils::Argmax;
use neural_network::{initializer, loss, optimizers, regularizer};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read};

fn with_model() {
    let mnist = Mnist::new();

    let seq = sequential![
        layers::Dense::new_with_regularizers::<initializer::He>(
            784,
            1024,
            Some(Box::new(regularizer::L2::default())),
            Some(Box::new(regularizer::L2::default())),
        ),
        activations::ReLU::default(),
        layers::Dropout::new(0.1),
        layers::Dense::new_with_regularizers::<initializer::He>(
            1024,
            512,
            Some(Box::new(regularizer::L2::default())),
            Some(Box::new(regularizer::L2::default())),
        ),
        activations::ReLU::default(),
        layers::Dropout::new(0.1),
        layers::Dense::new_with_regularizers::<initializer::He>(
            512,
            256,
            Some(Box::new(regularizer::L2::default())),
            Some(Box::new(regularizer::L2::default())),
        ),
        activations::ReLU::default(),
        layers::Dropout::new(0.1),
        layers::Dense::new_with_regularizers::<initializer::Xavier>(
            256,
            10,
            Some(Box::new(regularizer::L2::default())),
            Some(Box::new(regularizer::L2::default())),
        ),
        activations::Softmax::default(),
    ];

    let loss = loss::SparseCategoricalCrossEntropy::new(1e-7);
    let optim = optimizers::Adam::new(0.0005, 1e-5, 1e-7, 0.9, 0.999);
    let mut model = seq.compile(loss, optim);

    model.fit(&mnist.train_dataset, 300, 64, true, 100);
    let (pred, test_loss) = model.evaluate(&mnist.test_dataset);

    println!("Test Loss: {:?}", test_loss);

    println!(
        "Real labels: {:?}",
        mnist.test_dataset.outputs().slice(s![0..10])
    );
    println!(
        "Prediction : {:?}",
        pred.slice(s![0..10, ..]).argmax(Axis(1))
    );
}

fn main() {
    with_model()
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
