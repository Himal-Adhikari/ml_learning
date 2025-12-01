use std::collections::HashMap;

use nalgebra::{Const, Dyn, Matrix, VecStorage, ViewStorageMut};
use rand::prelude::*;

pub struct TestTrainSplit {
    pub training_set: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    pub test_split: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
}

pub fn stratified_shuffle_split(
    main_mat: &Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    strat_basis: &[u8],
    ratio: f64,
) -> TestTrainSplit {
    let mut rng = rand::rng();
    let num_rows = main_mat.nrows();

    let mut sequence = (0..num_rows).collect::<Vec<_>>();
    sequence.shuffle(&mut rng);

    let mut item_list_count = HashMap::new();
    for &item in strat_basis.iter() {
        match item_list_count.get_key_value(&item) {
            Some((_, &count)) => {
                item_list_count.insert(item, count + 1);
            }
            None => {
                item_list_count.insert(item, 1);
            }
        }
    }
    for (_, count) in item_list_count.iter_mut() {
        *count = (*count as f64 * ratio) as i32;
    }

    let mut counts = item_list_count.keys().map(|a| (*a, 0)).collect::<Vec<_>>();
    let mut training_idxs = Vec::new();
    let mut test_idxs = Vec::new();
    for &idx in sequence.iter() {
        for ((counts_val, counts_count), (_, list_count)) in
            counts.iter_mut().zip(item_list_count.iter())
        {
            if strat_basis[idx] == *counts_val {
                if *counts_count >= *list_count {
                    test_idxs.push(idx);
                } else {
                    training_idxs.push(idx);
                    *counts_count += 1;
                }
            }
        }
    }

    let training_mat = main_mat.select_rows(training_idxs.iter());
    let test_mat = main_mat.select_rows(test_idxs.iter());

    TestTrainSplit {
        training_set: training_mat,
        test_split: test_mat,
    }
}

pub fn std_scalar(
    mut mat: Matrix<f64, Dyn, Dyn, ViewStorageMut<'_, f64, Dyn, Dyn, Const<1>, Dyn>>,
) {
    let mean_std_vec = (0..mat.ncols())
        .map(|i| [mat.column(i).mean(), mat.column(i).variance().sqrt()])
        .collect::<Vec<_>>();

    for (idx, mean_std) in mean_std_vec.iter().enumerate() {
        for item in mat.column_mut(idx).iter_mut() {
            *item = (*item - mean_std[0]) / mean_std[1];
        }
    }
}
