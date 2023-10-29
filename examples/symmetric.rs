#![allow(non_snake_case)]
//! A completely symmetrical example, where the bounds describe the unit cube.
//! Due to the symmetry you can verify by hand that the unit vector is the optimal solution.

use approx::assert_abs_diff_eq;
use lp::prelude::*;
use ndarray::prelude::*;

fn main() {
    let problem_size = 1_000;

    let A_ub = &(Array2::<f64>::eye(problem_size) - 1.0) * -1.0;
    let b_ub = Array1::ones(problem_size) * (problem_size - 1) as f64;
    let c = -1.0 * Array1::ones(problem_size);

    let problem = Problem::target(&c).ub(&A_ub, &b_ub).build().unwrap();
    let solver = InteriorPoint::custom().disp(true).build().unwrap();

    let solution = solver.solve(&problem).unwrap();

    println!("solution found, minimal cost: {}", solution.fun());
    println!("required number of iterations: {}", solution.iteration());

    let expected = Array1::ones(problem_size);
    assert_abs_diff_eq!(solution.x(), &expected, epsilon = 1e-10);
}
