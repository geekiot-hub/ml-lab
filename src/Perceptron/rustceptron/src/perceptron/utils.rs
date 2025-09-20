use std::iter::zip;

// Dot product of two vectors
pub fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("The scalar product operation if not defined for vectors of different lengths!");
    }

    let mut result: f64 = 0.0;
    for (i, j) in zip(a, b) {
        result += i * j;
    }

    result
}
