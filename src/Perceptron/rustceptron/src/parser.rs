use std::fs::File;

use csv::StringRecord;

pub fn get_separated_iris_data(
    file_path: &str,
) -> (Vec<Vec<f64>>, Vec<i8>, Vec<Vec<f64>>, Vec<i8>) {
    let raw_data = get_raw_data_from_csv(file_path);

    let (features_vectors, targets) = get_iris_data(raw_data);

    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    let mut x_final = Vec::new();
    let mut y_final = Vec::new();

    let mut final_data_idxs: [u8; 99] = [0; 99];

    for idx in 25..75 {
        final_data_idxs[idx] = 1;
    }

    for (idx, &is_final) in final_data_idxs.iter().enumerate() {
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

    (x_train, y_train, x_final, y_final)
}

fn get_iris_data(raw_data: Vec<StringRecord>) -> (Vec<Vec<f64>>, Vec<i8>) {
    let mut features_vectors: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<i8> = Vec::new();

    for row in raw_data {
        let mut features: Vec<f64> = Vec::new();

        for (idx, cell_data) in row.iter().enumerate() {
            if idx == 0 || idx == 2 {
                features.push(match cell_data.parse() {
                    Ok(n) => n,
                    Err(err) => panic!("{}", err),
                });
            } else if idx == 4 {
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

fn get_raw_data_from_csv(file_path: &str) -> Vec<StringRecord> {
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
