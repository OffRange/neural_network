use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis};
use rand::prelude::SliceRandom;

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
}