use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Dimension, Ix1, Ix2, RemoveAxis};
use plotters::prelude::*;
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use std::error::Error;
use std::fs::File;
use std::io;
use std::io::{BufReader, Read};

pub trait Dataset {
    type InType: Clone;
    type OutType: Clone;

    type InDim: Dimension + RemoveAxis;
    type OutDim: Dimension + RemoveAxis;

    fn len(&self) -> usize;

    fn inputs(&self) -> ArrayView<Self::InType, Self::InDim>;
    fn outputs(&self) -> ArrayView<Self::OutType, Self::OutDim>;

    fn batch_iter(&self, batch_size: usize, shuffle: bool) -> BatchIterator<Self> {
        let n_samples = self.len();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            indices.shuffle(&mut rand::rng());
        }

        BatchIterator {
            dataset: self,
            indices,
            batch_size,
            current_idx: 0,
        }
    }
}

pub struct NNDataset<I, O, ID, OD> {
    inputs: Array<I, ID>,
    outputs: Array<O, OD>,
}

impl<I, O, ID, OD> NNDataset<I, O, ID, OD>
where
    ID: Dimension,
    OD: Dimension,
{
    /// Create a new dataset with the given inputs and outputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - An `Array` containing the input data.
    /// * `outputs` - An `Array` containing the output data.
    ///
    /// # Returns
    ///
    /// * An `NNDataset` instance.
    ///
    /// # Panics
    ///
    /// Panics if the number of samples in `inputs` and `outputs` do not match.
    pub fn new(inputs: Array<I, ID>, outputs: Array<O, OD>) -> Self {
        assert_eq!(inputs.len_of(Axis(0)), outputs.len_of(Axis(0)), "Number of samples must match between inputs and outputs");
        Self { inputs, outputs }
    }

    pub fn new_from_vec(input_shape: ID, output_shape: OD, inputs: Vec<I>, outputs: Vec<O>) -> Self {
        let inputs = Array::from_shape_vec(input_shape, inputs).unwrap();
        let outputs = Array::from_shape_vec(output_shape, outputs).unwrap();
        Self::new(inputs, outputs)
    }
}

impl<I, O, ID, OD> Dataset for NNDataset<I, O, ID, OD>
where
    I: Clone,
    O: Clone,
    ID: Dimension + RemoveAxis,
    OD: Dimension + RemoveAxis,
{
    type InType = I;
    type OutType = O;
    type InDim = ID;
    type OutDim = OD;

    fn len(&self) -> usize {
        self.inputs.len_of(Axis(0))
    }

    fn inputs(&self) -> ArrayView<Self::InType, Self::InDim> {
        self.inputs.view()
    }

    fn outputs(&self) -> ArrayView<Self::OutType, Self::OutDim> {
        self.outputs.view()
    }
}

pub struct BatchIterator<'a, D>
where
    D: Dataset + ?Sized,
{
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a, D> Iterator for BatchIterator<'a, D>
where
    D: Dataset + ?Sized,
{
    type Item = (
        Array<D::InType, D::InDim>,
        Array<D::OutType, D::OutDim>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        self.current_idx = end_idx;

        let inputs = self.dataset.inputs()
            .select(Axis(0), batch_indices);

        let outputs = self.dataset.outputs()
            .select(Axis(0), batch_indices);

        Some((inputs, outputs))
    }
}

type MnistDataset = NNDataset<f64, usize, Ix2, Ix1>;
pub struct Mnist {
    pub train_dataset: MnistDataset,
    pub test_dataset: MnistDataset,
}

impl Mnist {
    pub fn new() -> Self {
        let train_img = Mnist::load_img("data/mnist/train-images.idx3-ubyte").unwrap();
        let train_lbl = Mnist::load_label("data/mnist/train-labels.idx1-ubyte").unwrap();

        let test_img = Mnist::load_img("data/mnist/t10k-images.idx3-ubyte").unwrap();
        let test_lbl = Mnist::load_label("data/mnist/t10k-labels.idx1-ubyte").unwrap();

        Self {
            train_dataset: NNDataset::new(train_img, train_lbl),
            test_dataset: NNDataset::new(test_img, test_lbl),
        }
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
                .mapv(|x| x as f64 / 255.0)
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

pub fn create_spiral_dataset(num_points: usize, num_classes: usize) -> (Array2<f64>, Array1<usize>) {
    let total_points = num_points * num_classes;
    // Initialize the dataset arrays.
    let mut x = Array2::<f64>::zeros((total_points, 2));
    let mut y = Array1::<usize>::zeros(total_points);

    let mut rng = rand::rng();
    // Create a normal distribution with mean 0 and standard deviation 0.2 for noise.
    let noise = Normal::new(0.0, 0.2).unwrap();

    // Loop over each class (spiral arm).
    for class in 0..num_classes {
        for i in 0..num_points {
            // Compute the index for the overall dataset.
            let ix = class * num_points + i;
            // r is the radius, scaled from 0 to 1.
            let r = i as f64 / num_points as f64;
            // t is the angle: each class gets its own base angle (4.0 * class),
            // plus a linear increase along the arm and some added noise.
            let t = class as f64 * 4.0 + (i as f64 / num_points as f64) * 4.0 + noise.sample(&mut rng);
            // Multiply the angle to control the number of windings.
            let theta = t * 2.5;

            // Compute the (x, y) coordinates.
            x[[ix, 0]] = r * theta.sin();
            x[[ix, 1]] = r * theta.cos();
            // Set the class label.
            y[ix] = class;
        }
    }
    (x, y)
}

pub fn display_spiral_dataset(
    x: &Array2<f64>,
    y: &Array1<usize>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a drawing area (800x600 pixels) and fill it with white.
    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build a chart with fixed x and y ranges.
    let mut chart = ChartBuilder::on(&root)
        .caption("Spiral Dataset", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    chart.configure_mesh().draw()?;

    // Define a set of colors to be used for different classes.
    let colors = vec![&RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &YELLOW];

    // Plot each data point as a small filled circle.
    for (idx, point) in x.outer_iter().enumerate() {
        let class = y[idx];
        let color = colors[class % colors.len()];
        chart.draw_series(std::iter::once(Circle::new(
            (point[0], point[1]),
            3,
            color.filled(),
        )))?;
    }

    // Ensure the drawing is rendered and saved.
    root.present()?;
    println!("Plot saved as {}", path);
    Ok(())
}

pub(crate) fn visualize_pred<F>(data: &Array2<f64>, labels: &Array1<usize>, mut model_predict: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(f64, f64) -> usize,
{
    // Create a drawing area.
    let root = BitMapBackend::new("decision_boundaries.png", (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Setup chart. Adjust the x/y ranges according to your data.
    let mut chart = ChartBuilder::on(&root)
        .caption("Decision Boundaries", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    chart.configure_mesh().draw()?;

    // Define grid resolution. More grid points means smoother boundaries.
    let resolution = 300;
    let x_min = -1.5;
    let x_max = 1.5;
    let y_min = -1.5;
    let y_max = 1.5;
    let x_step = (x_max - x_min) / resolution as f64;
    let y_step = (y_max - y_min) / resolution as f64;

    // Draw the background. For each grid cell, compute the predicted class and fill with a light color.
    for i in 0..resolution {
        for j in 0..resolution {
            let x_val = x_min + i as f64 * x_step;
            let y_val = y_min + j as f64 * y_step;
            let class = model_predict(x_val, y_val);
            let color = match class {
                0 => RGBColor(255, 200, 200), // light red
                1 => RGBColor(200, 255, 200), // light green
                2 => RGBColor(200, 200, 255), // light blue
                _ => RGBColor(255, 255, 255),
            };
            // Draw a rectangle for each grid cell.
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_val, y_val), (x_val + x_step, y_val + y_step)],
                color.filled(),
            )))?;
        }
    }

    // Overlay the original data points.
    // You can use a different color or shape to visualize the actual points.
    for (point, label) in data.outer_iter().zip(labels.iter()) {
        let x_point = point[0];
        let y_point = point[1];
        let point_color = match label {
            0 => RED,
            1 => GREEN,
            2 => BLUE,
            _ => BLACK,
        };
        chart.draw_series(std::iter::once(Circle::new(
            (x_point, y_point),
            3,
            point_color.filled(),
        )))?;
    }

    // Save the drawing.
    root.present()?;
    println!("Result has been saved to decision_boundaries.png");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Dataset, Mnist, NNDataset};
    use ndarray::{array, Axis};

    #[test]
    #[should_panic]
    fn test_nn_dataset_panics() {
        let inputs = array![[1.0, 2.0], [3.0, 4.0]];
        let outputs = array![1, 2, 3];

        NNDataset::new(inputs, outputs);
    }

    #[test]
    fn test_dataset_len() {
        let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let outputs = array![[0.0], [1.0], [0.0]];
        let dataset = NNDataset::new(inputs, outputs);
        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_batch_iterator_no_shuffle() {
        let inputs = array![[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0]];
        let outputs = array![[1.0],
                             [0.0],
                             [1.0],
                             [0.0]];
        let dataset = NNDataset::new(inputs.clone(), outputs.clone());

        // Use a batch size of 2 and disable shuffling for predictable order.
        let batch_iter = dataset.batch_iter(2, false);

        let input_chunks = inputs.axis_chunks_iter(Axis(0), 2);
        let output_chunks = outputs.axis_chunks_iter(Axis(0), 2);
        let zipped_chunks = input_chunks.zip(output_chunks);

        for ((i, (batch_inputs, batch_outputs)), (expected_in, expected_out)) in batch_iter.enumerate().zip(zipped_chunks) {
            assert_eq!(batch_inputs.shape(), &[2, 2]);
            assert_eq!(batch_outputs.shape(), &[2, 1]);

            // Check first sample in the batch.
            assert_eq!(batch_inputs, expected_in);
            assert_eq!(batch_outputs, expected_out);
        }
    }

    #[test]
    fn test_batch_iterator_shuffle() {
        let inputs = array![[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0]];
        let outputs = array![[1.0],
                             [0.0],
                             [1.0],
                             [0.0]];
        let dataset = NNDataset::new(inputs, outputs);

        // Use a batch size of 3 and enable shuffling.
        let mut batch_iter = dataset.batch_iter(3, true);
        let mut total_samples = 0;

        while let Some((batch_inputs, _)) = batch_iter.next() {
            total_samples += batch_inputs.shape()[0];
            // The batch should have at most 3 samples.
            assert!(batch_inputs.shape()[0] <= 3);
        }

        // Verify that we have iterated over all samples.
        assert_eq!(total_samples, dataset.len());
    }

    #[test]
    fn test_mnist_loader() {
        let mnist = Mnist::new();

        assert_eq!(mnist.train_dataset.len(), 60_000);
        assert_eq!(mnist.test_dataset.len(), 10_000);
    }
}