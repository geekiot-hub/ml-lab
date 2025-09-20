mod parser;
mod perceptron;
use perceptron::Perceptron;

use std::iter::zip;

fn main() {
    // Get train and final data
    let (x_train, y_train, x_final, y_final) = parser::get_separated_iris_data("../../data.csv");

    // Perceptron settings
    let eta = 0.9;
    let n_iter = 10;
    let w_cnt = x_train
        .get(0)
        .expect("Can't get zero index element from features_vectors!")
        .len() as u16;

    // Create perceptron
    let mut perceptron = Perceptron::new(w_cnt, eta, n_iter);

    // Fit perceptron
    println!("{:#?}", perceptron);
    println!("Training perceptron...");
    perceptron.fit(&x_train, &y_train);
    println!("Successfully!");
    println!("{:#?}", perceptron);

    // Final check perceptron
    println!("Final check of the perceptron...");
    let mut final_errors = 0;

    for (features, target) in zip(x_final, y_final) {
        if perceptron.predict(&features) != target {
            final_errors += 1;
        }
    }
    println!("Successfully!");
    println!("Errors: {}", final_errors);
}
