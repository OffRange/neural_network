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
use ndarray::{Array2, ArrayBase, Ix2};
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

/// Computes the Jacobian matrix (i.e., the derivative) of the softmax function for a given output vector.
///
/// For a softmax output S, where `S[i][j]` is the probability of the j-th class for the i-th sample,
/// the derivative with respect to the input z is given by:
///
///     ∂S[i][j] / ∂z[i][k] = S[i][j] * (δ[j][k] - S[i][k])
///
/// Here, `δ[j][k]` is the Kronecker delta, which equals 1 if j = k and 0 otherwise.
///
/// In matrix form, the Jacobian can be expressed as:
///
///     J = diag(S) - S · Sᵀ
///
/// where:
/// - diag(S) is a diagonal matrix with the softmax outputs along the diagonal,
/// - S · Sᵀ represents the outer product of the softmax vector with itself.
///
/// # Arguments
///
/// * `softmax_out` - A 2D array (`ArrayBase<S, Ix2>`) containing the softmax outputs. Each row corresponds to a sample,
///   and each sample is represented as a column vector (one entry per row).
///
/// # Returns
///
/// Returns an `Array2<f64>` representing the computed Jacobian matrix of the softmax function,
/// which is particularly useful for backpropagation in neural networks.
fn jacobian_matrix<S>(softmax_out: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: ndarray::Data<Elem=f64>,
{
    let mut jacobian_matrix = Array2::<f64>::zeros((softmax_out.len(), softmax_out.len()));

    // First part of the softmax derivative (S_ij * δ_jk)
    for i in 0..jacobian_matrix.nrows() {
        jacobian_matrix[[i, i]] = softmax_out[[i, 0]];
    }

    // Second part of the softmax derivative: subtracting S_ij * S_ik
    // Because of the dot product, the input must be a row vector
    jacobian_matrix = jacobian_matrix - softmax_out.dot(&softmax_out.t());

    jacobian_matrix
}
