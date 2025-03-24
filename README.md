## Neural Network (But in Rust, Because Why Not?)

Ever wake up thinking, "Hey, today I'll reinvent neural networks from scratch in Rust!"? No? Well, too late—I did it
already. Welcome to my whimsical Rusty journey into neural network magic, inspired heavily by the
fantastic [NNFS book](https://nnfs.io/) by Harrison Kinsley and Daniel Kukiela.

### Seriously, Read This First

> [!WARNING]
> **DISCLAIMER:**
> This neural network is purely educational. It's like those IKEA instructions—you'll learn something new but probably
> shouldn't trust it to hold up your bookshelf. This isn't meant for production use unless you're into living
> dangerously.

My goals are simple:

- Understand neural networks and machine learning deeply.
- Learn Rust (actually one of my first rust project).

### How to Build

If you're adventurous (or just incredibly bored), you can add this crate directly from GitHub with:

```shell
cargo add --git https://github.com/OffRange/neural_network.git
```

Yes, someone might actually use this. Stranger things have happened.

You may also need to install `libfontconfig1-dev` as it is used by the plotters crate in some examples:

```shell
sudo apt install libfontconfig1-dev
```

### Usage

Below is a basic example demonstrating how to create, compile, train, and evaluate a neural network using this crate:

```rust
fn my_neural_network() {
    // Initialize your training and test datasets with input data and corresponding labels.
    let train_dataset = NNDataset::new(todo!("Input training data"), todo!("Input training labels"));
    let test_dataset = NNDataset::new(todo!("Input test data"), todo!("Input test labels"));

    // Define a sequential model with three layers.
    let model = sequential![
        // Input layer: transforms 2 inputs to 64 outputs using the He initializer.
        Dense::new::<initializers::He>(2, 64),
        ReLU::default(),

        // Hidden layer: further transforms 64 inputs to 64 outputs.
        Dense::new::<initializers::He>(64, 64),
        ReLU::default(),

        // Output layer: transforms 64 inputs to 10 outputs using the Xavier initializer,
        // followed by a softmax activation for multi-class classification.
        Dense::new::<initializers::Xavier>(64, 10),
        Softmax::default(),
    ];

    // Configure the optimizer and loss function.
    let optimizer = Adam::default();
    let loss = CategoricalCrossentropy::default();

    // Compile the model.
    let mut model = model.compile(optimizer, loss);

    // Train the model using the training dataset.
    model.fit(
        /* dataset = */ &train_dataset,
        /* epochs = */ 300,
        /* batch_size = */ 64,
        /* shuffle = */ true,
        /* print_every = */ 100,
        /* metrics = */ &[Box::new(MultiClassAccuracy)],
    );

    // Evaluate the model using the test dataset.
    let (predictions, test_loss) = model.evaluate(&test_dataset);
}
```

For additional examples and more detailed usage, please check out the [examples](examples) directory.

### Contributing

Contributions from researchers, practitioners, and enthusiasts are highly encouraged. Constructive criticism,
suggestions, or proposed enhancements are welcome—please open an issue or submit a pull request to facilitate
discussion.

---

*Happy Rusting!*

