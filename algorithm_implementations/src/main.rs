use nalgebra::DMatrix;
mod utils;
use utils::*;

#[derive(Debug, serde::Deserialize)]
struct Iris {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    species: String,
}

fn main() {
    let output_iter = read_csv_to_struct();
    let instance_num = output_iter.len();
    let class_num = 3;
    let feature_num = 4;
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
        output_iter
            .iter()
            .map(|data| {
                {
                    [
                        data.sepal_length,
                        data.sepal_width,
                        data.petal_length,
                        data.petal_width,
                        (data.species == "setosa") as u8 as f32,
                        (data.species == "versicolor") as u8 as f32,
                        (data.species == "virginica") as u8 as f32,
                    ]
                    .into_iter()
                }
            })
            .flatten(),
    );
    let theta_mat = DMatrix::<f32>::zeros(class_num, feature_num);
    stratified_shuffle_split(&comb_mat, strat_basis, 0.8);
}

fn read_csv_to_struct() -> Vec<Iris> {
    let mut rdr = csv::Reader::from_path("./iris.csv").unwrap();
    let mut data: Vec<Iris> = Vec::new();
    for result in rdr.deserialize() {
        data.push(result.unwrap());
    }
    data
}
