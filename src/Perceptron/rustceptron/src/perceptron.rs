mod utils;
use utils::dot;

use std::iter::zip;

#[derive(Debug)]
pub struct Perceptron {
    w: Vec<f64>,
    b: f64,
    eta: f64,
    n_iter: u16,
    train_errors: Vec<u16>,
}

impl Perceptron {
    pub fn new(w_count: u16, eta: f64, n_iter: u16) -> Perceptron {
        if w_count < 1 {
            panic!("The dimension of the features vector (w_count) must be greater than 1!");
        }

        let w: Vec<f64> = vec![0.1].repeat(w_count.into());

        Perceptron {
            w: w,
            b: 0.0,
            eta: eta,
            n_iter: n_iter,
            train_errors: Vec::new(),
        }
    }

    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<i8>) {
        for _ in 0..self.n_iter {
            let mut current_epoch_errors = 0;

            for (features, target) in zip(x_train, y_train) {
                // Verification of the perceptron's answer
                let update = target - self.predict(features);

                if update == 0 {
                    continue;
                }

                // Training in case of the perceptron's incorrect answer
                let update = update as f64;

                self.b += self.eta * update;

                for (feature, weight) in zip(features, &mut self.w) {
                    *weight += self.eta * update * feature;
                }

                current_epoch_errors += 1;
            }

            self.train_errors.push(current_epoch_errors);
        }
    }

    fn get_net_input(&self, features: &Vec<f64>) -> f64 {
        return dot(features, &self.w) + self.b;
    }

    pub fn predict(&self, features: &Vec<f64>) -> i8 {
        let net_input = self.get_net_input(features);

        if net_input >= 0.0 {
            1
        } else {
            0
        }
    }
}
