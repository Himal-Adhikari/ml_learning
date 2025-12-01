use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage, ViewStorage};

pub struct SoftMax {
    parameter_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    prediction_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    eta: f64,
}

impl SoftMax {
    pub fn new(nrows: usize, ncols: usize, eta: f64) -> Self {
        SoftMax {
            parameter_matrix: DMatrix::zeros(nrows, ncols),
            prediction_matrix: DMatrix::zeros(0, 0),
            eta,
        }
    }
    pub fn fit(
        &mut self,
        x_mat: Matrix<f64, Dyn, Dyn, ViewStorage<f64, Dyn, Dyn, Const<1>, Dyn>>,
        y_mat: Matrix<f64, Dyn, Dyn, ViewStorage<f64, Dyn, Dyn, Const<1>, Dyn>>,
    ) {
        let num_iterations = 1000;

        for _ in 0..num_iterations {
            let gradient_vector = self.gradient_vector(x_mat, y_mat);
            self.parameter_matrix -= self.eta * gradient_vector;
        }
        // println!("{}", self.parameter_matrix);
    }

    pub fn predict(
        &mut self,
        x_mat: Matrix<f64, Dyn, Dyn, ViewStorage<f64, Dyn, Dyn, Const<1>, Dyn>>,
    ) -> Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> {
        let mut prediction_mat =
            DMatrix::<f64>::zeros(x_mat.nrows(), self.parameter_matrix.nrows());
        for i in 0..x_mat.nrows() {
            let proba_mat = self.proba(&x_mat.row(i));
            let (max_val_idx, _) = proba_mat
                .iter()
                .enumerate()
                .max_by(|(_, val1), (_, val2)| val1.partial_cmp(val2).unwrap())
                .unwrap();
            prediction_mat[(i, max_val_idx)] = 1.0;
        }
        self.prediction_matrix = prediction_mat.clone();
        prediction_mat
    }

    pub fn accuracy(
        &self,
        y_mat: Matrix<f64, Dyn, Dyn, ViewStorage<f64, Dyn, Dyn, Const<1>, Dyn>>,
    ) -> f64 {
        1.0 - y_mat
            .iter()
            .zip(self.prediction_matrix.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / (2.0 * y_mat.nrows() as f64)
    }

    fn proba(
        &self,
        x_mat: &Matrix<f64, Const<1>, Dyn, ViewStorage<'_, f64, Const<1>, Dyn, Const<1>, Dyn>>,
    ) -> Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> {
        let scores = (0..self.parameter_matrix.nrows())
            .map(|i| x_mat * self.parameter_matrix.row(i).transpose())
            .map(|x| x[(0, 0)])
            .map(|num| num.exp())
            .collect::<Vec<_>>();
        let sum = scores.iter().sum::<f64>();
        DMatrix::from_iterator(
            1,
            self.parameter_matrix.nrows(),
            scores.into_iter().map(|x| x / sum),
        )
    }

    fn gradient_vector(
        &self,
        x_mat: Matrix<f64, Dyn, Dyn, ViewStorage<f64, Dyn, Dyn, Const<1>, Dyn>>,
        y_mat: Matrix<f64, Dyn, Dyn, ViewStorage<f64, Dyn, Dyn, Const<1>, Dyn>>,
    ) -> Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> {
        let mut grad_vec = (0..x_mat.nrows())
            .map(|idx| (self.proba(&x_mat.row(idx)) - y_mat.row(idx)).transpose() * x_mat.row(idx))
            .sum::<Matrix<_, _, _, _>>();
        let total_samples = x_mat.nrows() as f64;
        for item in grad_vec.iter_mut() {
            *item /= total_samples;
        }
        grad_vec
    }
}
