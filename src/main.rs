mod layer;

use ndarray::array;

fn main() {
    // Values based on the nnfs book from Harrison Kinsley

    // n_batch x n_in
    let input = array![
        [1., 2., 3., 2.5],
        [2., 5., -1., 2.],
        [-1.5, 2.7, 3.3, -0.8]
    ];

    // n_neurons x n_in
    let _weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ];

    // n_in x n_neurons
    // This is the transpose of the weights' matrix.
    // Maybe this is faster than using the weights matrix directly, as it avoids the transpose
    // operation on each training epoch.
    // Or maybe it isn't faster, as the transpose operation only reverses the dimensions
    // and the strides.
    let weights_t = array![
        [0.2, 0.5, -0.26],
        [0.8, -0.91, -0.27],
        [-0.5, 0.26, 0.17],
        [1.0, -0.5, 0.87]
    ];

    let biases = array![2., 3., 0.5];

    let r = input.dot(&weights_t.view().reversed_axes()) + &biases;
    println!("{:?}", r);
}
