use std::fs::File;

use csv::StringRecord;

pub fn get_iris_data(raw_data: Vec<StringRecord>) -> (Vec<Vec<f64>>, Vec<i64>) {
    let mut features_vectors: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<i64> = Vec::new();

    for row in raw_data {
        let mut features: Vec<f64> = Vec::new();

        for (idx, cell_data) in row.iter().enumerate() {
            if idx == 0 {
                features.push(match cell_data.parse() {
                    Ok(n) => n,
                    Err(err) => panic!("{}", err),
                });
            }

            if idx == 2 {
                features.push(match cell_data.parse() {
                    Ok(n) => n,
                    Err(err) => panic!("{}", err),
                })
            }

            if idx == 4 {
                if cell_data == "Iris-setosa" {
                    targets.push(1);
                } else {
                    targets.push(0);
                }
            }
        }

        features_vectors.push(features);
    }

    (features_vectors, targets)
}

pub fn get_raw_data_from_csv(file_path: &str) -> Vec<StringRecord> {
    let file = match File::open(file_path) {
        Ok(f) => f,
        Err(err) => {
            panic!("Can't read file from path {}\nError: {}", file_path, err);
        }
    };

    let mut reader = csv::Reader::from_reader(file);

    let mut raw_data: Vec<StringRecord> = Vec::new();

    for record in reader.records() {
        match record {
            Ok(rec) => {
                raw_data.push(rec);
            }
            Err(err) => panic!("{}", err),
        }
    }

    raw_data
}
