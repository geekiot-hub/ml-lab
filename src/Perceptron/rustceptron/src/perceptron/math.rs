use std::iter::zip;

pub fn scalar_mult(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut result: f64 = 0.0;
    for (i, j) in zip(a, b) {
        result += i * j;
    }

    result
}
