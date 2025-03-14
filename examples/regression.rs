use ndarray::{Array1, Array2, Axis, Ix1, stack};
use neural_network::Module;
use neural_network::data::{Dataset, NNDataset};
use neural_network::loss::Loss;
use neural_network::metric::{Metric, RegressionAccuracy, StdTolerance};
use neural_network::module::activations;
use neural_network::module::layers::{Dense, TrainableLayer};
use neural_network::optimizers::Optimizer;
use neural_network::{initializer, loss, optimizers, regularizer};
use plotters::prelude::*;
use rand::distr::Distribution;
use rand_distr::Normal;
use std::error::Error;
use std::f64::consts::TAU;

fn main() {
    let SineDatasets {
        train_dataset,
        test_dataset,
        validation_dataset,
    } = generate_sine_data(1_000, 0.0, TAU);

    // Define Model
    let mut layer1 = Dense::new_with_regularizers::<initializer::He>(
        1,
        64,
        Some(Box::new(regularizer::L2::new(5e-4))),
        None,
    );
    let mut activation1 = activations::ReLU::default();

    let mut layer2 = Dense::new_with_regularizers::<initializer::He>(
        64,
        128,
        Some(Box::new(regularizer::L2::new(5e-4))),
        None,
    );
    let mut activation2 = activations::ReLU::default();

    let mut layer3 = Dense::new_with_regularizers::<initializer::He>(
        128,
        1,
        Some(Box::new(regularizer::L2::new(5e-4))),
        None,
    );
    let mut activation3 = activations::Linear;

    let loss = loss::MeanSquaredError;
    let mut optimizer = optimizers::Adam::new(0.0005, 1e-3, 1e-7, 0.9, 0.999);

    macro_rules! forward {
        (&$x:ident) => {{
            let layer1_out = layer1.forward(&$x);
            let activation1_out = activation1.forward(&layer1_out);

            let layer2_out = layer2.forward(&activation1_out);
            let activation2_out = activation2.forward(&layer2_out);

            let layer3_out = layer3.forward(&activation2_out);
            activation3.forward(&layer3_out)
        }};
    }

    // Train Model
    let epochs = 10_000;
    let batch_size = 32;
    for epoch in 0..epochs {
        let mut loss_val = 0.0;
        let mut accuracy_val = 0.0;

        optimizer.pre_update();
        for (train_x, train_y) in train_dataset.batch_iter(batch_size, true) {
            let train_x = train_x.insert_axis(Axis(1));
            let train_y = train_y.insert_axis(Axis(1));

            let y_pred = forward!(&train_x);

            // Loss
            let regularization_loss = layer1.regularization_losses().0
                + layer2.regularization_losses().0
                + layer3.regularization_losses().0;
            loss_val += loss.calculate(&y_pred, &train_y) + regularization_loss;

            let acc = RegressionAccuracy::new(StdTolerance::new(train_y.view(), 0., 250.));
            accuracy_val += acc.evaluate(&y_pred, &train_y);

            // Backward
            let d_values = loss.backwards(&y_pred, &train_y);
            let d_values = activation3.backward(&d_values);
            let d_values = layer3.backward(&d_values);

            let d_values = activation2.backward(&d_values);
            let d_values = layer2.backward(&d_values);

            let d_values = activation1.backward(&d_values);
            let _ = layer1.backward(&d_values);

            // Update parameters
            optimizer.update(&mut layer1);
            optimizer.update(&mut layer2);
            optimizer.update(&mut layer3);
        }

        if epoch % 100 == 0 {
            println!(
                "Epoch: {}, Loss: {:.4}, Accuracy: {:.4}",
                epoch,
                loss_val / batch_size as f64,
                accuracy_val / batch_size as f64
            );
        }
    }

    let data = test_dataset.inputs().to_owned().insert_axis(Axis(1));
    let prediction: Array2<f64> = forward!(&data);
    let prediction = prediction.remove_axis(Axis(1));

    let validation_data = stack![
        Axis(1),
        validation_dataset.inputs(),
        validation_dataset.outputs()
    ];

    display_sine_data(
        "predicted_sine.png",
        &validation_data,
        &test_dataset.outputs().to_owned(),
        Some(&prediction),
    )
    .expect("failed to display sine_data");
}

struct SineDatasets {
    train_dataset: NNDataset<f64, f64, Ix1, Ix1>,
    test_dataset: NNDataset<f64, f64, Ix1, Ix1>,
    validation_dataset: NNDataset<f64, f64, Ix1, Ix1>,
}

fn generate_sine_data(samples: usize, start: f64, end: f64) -> SineDatasets {
    let sine_x = Array1::linspace(start, end, samples);
    let sin_y = sine_x.sin();

    let normal = Normal::new(0., 0.2).unwrap();
    let mut rng = rand::rng();

    let noise = Array1::from_iter((0..samples).map(|_| normal.sample(&mut rng)));
    let train_dataset = NNDataset::new(sine_x.clone(), sin_y.clone() + &noise);

    let noise = Array1::from_iter((0..samples).map(|_| normal.sample(&mut rng)));
    let test_dataset = NNDataset::new(sine_x.clone(), sin_y.clone() + &noise);

    let validation_dataset = NNDataset::new(sine_x, sin_y);

    SineDatasets {
        train_dataset,
        test_dataset,
        validation_dataset,
    }
}

fn display_sine_data(
    path: &str,
    sine_data: &Array2<f64>,
    test_data: &Array1<f64>,
    predicted_data: Option<&Array1<f64>>,
) -> Result<(), Box<dyn Error>> {
    if sine_data.shape()[1] != 2 {
        return Err("sine_data must have exactly two columns".into());
    }

    if let Some(pred) = predicted_data {
        if pred.len() != sine_data.shape()[0] {
            return Err("predicted_data length must match the number of rows in sine_data".into());
        }
    }

    let root_area = BitMapBackend::new(path, (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Sine Wave", ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..TAU, -1.5..1.5)?;

    chart
        .configure_mesh()
        .x_labels(12)
        .y_labels(12)
        .label_style(("sans-serif", 30))
        .axis_desc_style(("sans-serif", 30))
        .draw()?;

    let legend_element = |color: RGBColor| {
        move |(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&color).stroke_width(2),
            )
        }
    };

    chart
        .draw_series(LineSeries::new(
            sine_data.axis_iter(Axis(0)).map(|row| (row[0], row[1])),
            ShapeStyle::from(&RED).stroke_width(2),
        ))?
        .label("Sine Wave")
        .legend(legend_element(RED));

    chart
        .draw_series(LineSeries::new(
            sine_data
                .axis_iter(Axis(0))
                .zip(test_data.iter())
                .map(|(row, &pred)| (row[0], pred)),
            &GREEN,
        ))?
        .label("Test Input")
        .legend(legend_element(GREEN));

    if let Some(pred_data) = predicted_data {
        chart
            .draw_series(LineSeries::new(
                sine_data
                    .axis_iter(Axis(0))
                    .zip(pred_data.iter())
                    .map(|(row, &pred)| (row[0], pred)),
                &BLUE,
            ))?
            .label("Predicted Sine Wave")
            .legend(legend_element(BLUE));
    }

    chart
        .configure_series_labels()
        .label_font(("sans-serif", 25))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    Ok(())
}
