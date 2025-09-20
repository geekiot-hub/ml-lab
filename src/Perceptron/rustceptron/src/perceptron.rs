use std::iter::zip;

mod math;

#[derive(Debug)]
pub struct Perceptron {
    w: Vec<f64>,
    b: f64,
    eta: f64,
    n_iter: u8,
    train_errors: Vec<u8>,
}

impl Perceptron {
    pub fn new(w_count: u8, eta: f64, n_iter: u8) -> Perceptron {
        let mut w: Vec<f64> = Vec::new();

        for _ in 0..w_count {
            w.push(0.1);
        }

        Perceptron {
            w: w,
            b: 0.0,
            eta: eta,
            n_iter: n_iter,
            train_errors: Vec::new(),
        }
    }

    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<i64>) {
        for _ in 0..self.n_iter {
            let mut errors_cnt = 0;

            for (features, target) in zip(x_train, y_train) {
                let update = target - self.predict(features);

                if update == 0 {
                    continue;
                }

                let update = update as f64;

                self.b += update * self.eta;

                for (feature, weight) in zip(features, &mut self.w) {
                    *weight += update * feature * self.eta;
                }

                errors_cnt += 1;
            }

            self.train_errors.push(errors_cnt);
        }
    }

    fn get_net_input(&self, features: &Vec<f64>) -> f64 {
        return math::scalar_mult(features, &self.w) + self.b;
    }

    pub fn predict(&self, features: &Vec<f64>) -> i64 {
        let net_input = self.get_net_input(features);

        if net_input >= 0.0 {
            1
        } else {
            0
        }
    }
}
