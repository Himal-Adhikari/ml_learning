use std::collections::HashMap;

use nalgebra::{Dyn, Matrix, VecStorage};
use rand::prelude::*;

pub fn stratified_shuffle_split(
    main_mat: &Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>,
    strat_basis: Vec<u8>,
    ratio: f32,
) -> Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>> {
    // (Item, Count)
    let mut rng = rand::rng();
    let num_rows = main_mat.nrows();

    let mut sequence = (0..num_rows).into_iter().collect::<Vec<_>>();
    sequence.shuffle(&mut rng);

    let mut item_list = HashMap::new();
    for &item in strat_basis.iter() {
        match item_list.get_key_value(&item) {
            Some((_, &count)) => {
                item_list.insert(item, count + 1);
            }
            None => {
                item_list.insert(item, 1);
            }
        }
    }
    for (_, count) in item_list.iter_mut() {
        *count = (*count as f32 * ratio) as i32;
    }

    let mut counts = item_list.iter().map(|(a, _)| (*a, 0)).collect::<Vec<_>>();
    let mut training_idxs = Vec::new();
    let mut test_idxs = Vec::new();
    for &idx in sequence.iter() {
        for ((counts_val, counts_count), (_, list_count)) in counts.iter_mut().zip(item_list.iter())
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

    println!("{}", training_mat);
    println!("{}", test_mat);

    todo!()
}
