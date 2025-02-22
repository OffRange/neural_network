use ndarray::{Array1, Array2};
use plotters::prelude::*;
use rand::prelude::Distribution;
use rand_distr::Normal;

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
