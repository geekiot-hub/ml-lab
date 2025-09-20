mod data;
mod perceptron;

use std::iter::zip;

use perceptron::Perceptron;

fn main() {
    let raw_data = data::get_raw_data_from_csv("../../data.csv");

    let (features_vectors, targets) = data::get_iris_data(raw_data);

    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<i64> = Vec::new();

    let mut x_final = x_train.clone();
    let mut y_final = y_train.clone();

    let mut idx_to_final: [u8; 99] = [0; 99];

    for idx in 25..75 {
        idx_to_final[idx] = 1;
    }

    for (idx, &is_final) in idx_to_final.iter().enumerate() {
        if is_final == 1 {
            x_final.push(
                features_vectors
                    .get(idx)
                    .expect("Can't get features from vector")
                    .to_vec(),
            );

            y_final.push(*targets.get(idx).expect("Can't get target from vector"));
        } else {
            x_train.push(
                features_vectors
                    .get(idx)
                    .expect("Can't get features from vector")
                    .to_vec(),
            );
            y_train.push(*targets.get(idx).expect("Can't get target from vector"));
        }
    }

    let mut perceptron = Perceptron::new(
        features_vectors
            .get(0)
            .expect("Can't get zero-index-features vector from features_vectors")
            .len() as u8,
        0.9,
        10,
    );

    // Fit
    // println!("{:#?}", perceptron);
    println!("Training perceptron...");
    perceptron.fit(&x_train, &y_train);
    println!("Successfully!");
    // println!("{:#?}", perceptron);

    // Final check
    println!("Final check of the perceptron");
    let mut final_errors = 0;

    for (features, target) in zip(x_final, y_final) {
        if perceptron.predict(&features) != target {
            final_errors += 1;
        }
    }
    println!("Errors: {}", final_errors);
}
