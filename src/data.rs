/*!
Module for handling datasets.

This module provides a `Dataset` trait that can be implemented for any type that represents a
dataset. The `NNDataset` struct is provided as a concrete implementation of the `Dataset` trait
for neural network data.
*/

use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis, StrideShape};
use rand::prelude::SliceRandom;

/// The `Dataset` trait represents a collection of input-output pairs.
///
/// It provides methods to access the underlying input and output data as immutable
/// views, and it also provides a method to create an iterator over batches of samples.
///
/// # Associated Types
///
/// - `InType`: The element type of the input data.
/// - `OutType`: The element type of the output data.
/// - `InDim`: The dimensionality type for the input data (must implement `Dimension` and `RemoveAxis`).
/// - `OutDim`: The dimensionality type for the output data (must implement `Dimension` and `RemoveAxis`).
pub trait Dataset {
    type InType: Clone;
    type OutType: Clone;

    type InDim: Dimension + RemoveAxis;
    type OutDim: Dimension + RemoveAxis;

    /// Returns the number of samples in the dataset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let dataset = NNDataset::new(array![[1., 2.], [3., 4.]], array![[0.], [1.]]);
    ///
    /// assert_eq!(2, dataset.len());
    /// ```
    fn len(&self) -> usize;

    /// Checks if the dataset is empty. Returns `true` if the dataset is empty, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let dataset = NNDataset::new(array![[1., 2.], [3., 4.]], array![[0.], [1.]]);
    ///
    /// assert!(!dataset.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Provides a view of the input data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let outputs = array![[0.0], [1.0], [0.0]];
    /// let dataset = NNDataset::new(inputs.clone(), outputs);
    ///
    /// assert_eq!(dataset.inputs(), inputs.view());
    /// ```
    fn inputs(&self) -> ArrayView<Self::InType, Self::InDim>;

    /// Provides a view of the output data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let outputs = array![[0.0], [1.0], [0.0]];
    /// let dataset = NNDataset::new(inputs, outputs.clone());
    ///
    /// assert_eq!(dataset.outputs(), outputs.view());
    /// ```
    fn outputs(&self) -> ArrayView<Self::OutType, Self::OutDim>;

    /// Creates a [batch iterator](BatchIterator) over the dataset.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The number of samples in each batch
    /// * `shuffle` - Whether to randomly shuffle the data before creating batches
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    /// let outputs = array![[0.0], [1.0], [0.0], [1.0]];
    /// let dataset = NNDataset::new(inputs, outputs);
    ///
    /// // Iterate with a batch size of 2, without shuffling
    /// let mut batch_count = 0;
    /// for (batch_inputs, batch_outputs) in dataset.batch_iter(2, false) {
    ///     assert_eq!(batch_inputs.shape()[0], 2);
    ///     batch_count += 1;
    /// }
    /// assert_eq!(batch_count, 2);
    /// ```
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

/// A concrete implementation of a neural network dataset.
///
/// This struct stores input and output data as multidimensional arrays
/// and implements the [Dataset] trait.
///
/// # Type Parameters
///
/// - `I`: Input data element type
/// - `O`: Output data element type
/// - `ID`: Input data dimensionality
/// - `OD`: Output data dimensionality
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use neural_network::data::NNDataset;
///
/// // Create a dataset for a simple regression problem
/// let inputs = array![[1.0], [2.0], [3.0], [4.0]];
/// let outputs = array![[2.0], [4.0], [6.0], [8.0]];
/// let dataset = NNDataset::new(inputs, outputs);
///
/// // Create a dataset from vectors
/// let inputs_vec = vec![1.0, 2.0, 3.0, 4.0];
/// let outputs_vec = vec![2.0, 4.0, 6.0, 8.0];
/// let dataset_from_vec = NNDataset::new_from_vec(
///     (4, 1),
///     (4, 1),
///     inputs_vec,
///     outputs_vec
/// );
/// ```
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
    /// # Panics
    ///
    /// Panics if the number of samples in `inputs` and `outputs` do not match.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let inputs = array![[1.0, 2.0], [3.0, 4.0]];
    /// let outputs = array![[0.0], [1.0]];
    /// let dataset = NNDataset::new(inputs.clone(), outputs.clone());
    ///
    /// assert_eq!(inputs, dataset.inputs());
    /// assert_eq!(outputs, dataset.outputs());
    /// ```
    pub fn new(inputs: Array<I, ID>, outputs: Array<O, OD>) -> Self {
        assert_eq!(
            inputs.len_of(Axis(0)),
            outputs.len_of(Axis(0)),
            "Number of samples must match between inputs and outputs"
        );
        Self { inputs, outputs }
    }

    /// Create a new dataset from vectors with specified shapes.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input data
    /// * `output_shape` - The shape of the output data
    /// * `inputs` - A vector of input data elements
    /// * `outputs` - A vector of output data elements
    ///
    /// # Panics
    ///
    /// Panics if the vectors cannot be shaped into the specified dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use neural_network::data::{Dataset, NNDataset};
    ///
    /// let inputs_vec = vec![1.0, 2.0, 3.0, 4.0];
    /// let outputs_vec = vec![0.0, 1.0, 0.0, 1.0];
    ///
    /// let dataset = NNDataset::new_from_vec(
    ///     (4, 1),  // input shape
    ///     (4, 1),  // output shape
    ///     inputs_vec,
    ///     outputs_vec
    /// );
    ///
    /// assert_eq!(array![[1.0], [2.0], [3.0], [4.0]].view(), dataset.inputs());
    /// ```
    pub fn new_from_vec<InputSh, OutputSh>(
        input_shape: InputSh,
        output_shape: OutputSh,
        inputs: Vec<I>,
        outputs: Vec<O>,
    ) -> Self
    where
        InputSh: Into<StrideShape<ID>>,
        OutputSh: Into<StrideShape<OD>>,
    {
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

/// An iterator that generates batches from a dataset.
///
/// Supports configurable batch sizes and optional data shuffling.
///
/// # Type Parameters
///
/// - `D`: The type of dataset being iterated over
///
/// # Example
///
/// ```rust
/// use ndarray::{array, aview2};
/// use neural_network::data::{Dataset, NNDataset};
///
/// let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let outputs = array![[0.0], [1.0], [2.0], [3.0]];
/// let dataset = NNDataset::new(inputs, outputs);
///
/// // Iterate with a batch size of 2, without shuffling
/// for (batch_inputs, batch_outputs) in dataset.batch_iter(2, false) {
///     println!("Batch inputs shape: {:?}", batch_inputs.shape());
///     println!("Batch outputs shape: {:?}", batch_outputs.shape());
/// }
///
/// let mut batch_iter = dataset.batch_iter(2, false);
/// assert_eq!(Some((array![[1.0, 2.0], [3.0, 4.0]], array![[0.0], [1.0]])), batch_iter.next());
/// assert_eq!(Some((array![[5.0, 6.0], [7.0, 8.0]], array![[2.0], [3.0]])), batch_iter.next());
///
/// // Iterate with shuffling
/// for (batch_inputs, batch_outputs) in dataset.batch_iter(3, true) {
///     // Batches will be in random order
///     println!("Shuffled batch inputs shape: {:?}", batch_inputs.shape());
/// }
/// ```
pub struct BatchIterator<'a, D>
where
    D: Dataset + ?Sized,
{
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl<D> Iterator for BatchIterator<'_, D>
where
    D: Dataset + ?Sized,
{
    type Item = (Array<D::InType, D::InDim>, Array<D::OutType, D::OutDim>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        self.current_idx = end_idx;

        let inputs = self.dataset.inputs().select(Axis(0), batch_indices);

        let outputs = self.dataset.outputs().select(Axis(0), batch_indices);

        Some((inputs, outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::{Dataset, NNDataset};
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
        let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let outputs = array![[1.0], [0.0], [1.0], [0.0]];
        let dataset = NNDataset::new(inputs.clone(), outputs.clone());

        // Use a batch size of 2 and disable shuffling for predictable order.
        let batch_iter = dataset.batch_iter(2, false);

        let input_chunks = inputs.axis_chunks_iter(Axis(0), 2);
        let output_chunks = outputs.axis_chunks_iter(Axis(0), 2);
        let zipped_chunks = input_chunks.zip(output_chunks);

        for ((batch_inputs, batch_outputs), (expected_in, expected_out)) in
            batch_iter.zip(zipped_chunks)
        {
            assert_eq!(batch_inputs.shape(), &[2, 2]);
            assert_eq!(batch_outputs.shape(), &[2, 1]);

            // Check first sample in the batch.
            assert_eq!(batch_inputs, expected_in);
            assert_eq!(batch_outputs, expected_out);
        }
    }

    #[test]
    fn test_batch_iterator_shuffle() {
        let inputs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let outputs = array![[1.0], [0.0], [1.0], [0.0]];
        let dataset = NNDataset::new(inputs, outputs);

        // Use a batch size of 3 and enable shuffling.
        let batch_iter = dataset.batch_iter(3, true);
        let mut total_samples = 0;

        for (batch_inputs, _) in batch_iter {
            total_samples += batch_inputs.shape()[0];
            // The batch should have at most 3 samples.
            assert!(batch_inputs.shape()[0] <= 3);
        }

        // Verify that we have iterated over all samples.
        assert_eq!(total_samples, dataset.len());
    }
}
