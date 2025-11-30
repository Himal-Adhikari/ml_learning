use nalgebra::{Dyn, Matrix, VecStorage};
use rand::prelude::*;

pub fn stratified_shuffle_split(
    main_mat: &Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>,
    strat_basis: Vec<u8>,
    ratio: f32,
) -> Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>> {
    // (Item, Count)
    let mut item_list: Vec<(u8, u32)> = Vec::new();
    let mut rng = rand::rng();
    let num_rows = main_mat.nrows();

    let mut sequence = (0..num_rows).into_iter().collect::<Vec<_>>();
    sequence.shuffle(&mut rng);

    'outer: for &item in strat_basis.iter() {
        for i in item_list.iter_mut() {
            if i.0 == item {
                i.1 += 1;
                continue 'outer;
            }
        }
        item_list.push((item, 1));
    }

    for item in item_list.iter_mut() {
        item.1 = (item.1 as f32 * ratio) as u32;
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
