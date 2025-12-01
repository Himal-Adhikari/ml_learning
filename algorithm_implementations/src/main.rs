use nalgebra::DMatrix;
mod softmax;
mod utils;
use softmax::*;
use utils::*;

#[derive(Debug, serde::Deserialize)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

fn main() {
    let output_iter = read_csv_to_struct();
    let instance_num = output_iter.len();
    let class_num = 3;
    let feature_num = 4 + 1;
    let strat_basis = output_iter
        .iter()
        .map(|data| match data.species.as_str() {
            "setosa" => 0,
            "versicolor" => 1,
            "virginica" => 2,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    let comb_mat = DMatrix::from_row_iterator(
        instance_num,
        feature_num + class_num,
        output_iter.iter().flat_map(|data| {
            {
                [
                    1.0,
                    data.sepal_length,
                    data.sepal_width,
                    data.petal_length,
                    data.petal_width,
                    (data.species == "setosa") as u8 as f64,
                    (data.species == "versicolor") as u8 as f64,
                    (data.species == "virginica") as u8 as f64,
                ]
                .into_iter()
            }
        }),
    );
    let mut split_data = stratified_shuffle_split(&comb_mat, &strat_basis, 0.7);
    std_scalar(split_data.training_set.columns_mut(1, feature_num - 1));
    std_scalar(split_data.test_split.columns_mut(1, feature_num - 1));

    let mut softmax_regressor = SoftMax::new(class_num, feature_num, 0.05);
    softmax_regressor.fit(
        split_data.training_set.columns(0, feature_num),
        split_data.training_set.columns(feature_num, class_num),
    );

    let _ = softmax_regressor.predict(split_data.test_split.columns(0, feature_num));
    println!(
        "Accuracy = {}%",
        softmax_regressor.accuracy(split_data.test_split.columns(feature_num, class_num)) * 100.0
    );
}

fn read_csv_to_struct() -> Vec<Iris> {
    let mut rdr = csv::Reader::from_path("./iris.csv").unwrap();
    let mut data: Vec<Iris> = Vec::new();
    for result in rdr.deserialize() {
        data.push(result.unwrap());
    }
    data
}
