use std::fmt::Debug;

use crate::{
    error::LinearProgramError,
    linear_program::{OptimizeResult, Problem},
};

pub trait Solver<F: Debug> {
    fn solve(&self, problem: &Problem<F>) -> Result<OptimizeResult<F>, LinearProgramError<F>>;
}
