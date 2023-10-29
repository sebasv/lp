//! Solvers for linear programs.
pub mod interior_point;

pub use interior_point::InteriorPoint;
use ndarray::Array1;

use std::fmt::Debug;

use crate::{error::LinearProgramError, linear_program::Problem};

/// Solver trait that any solver should implement to make experimentation with different solvers more easy.
pub trait Solver<F: Debug> {
    /// Solve a linear programming problem. Returns a [`LinearProgramError`] error if the solver runs into problems.
    /// The possible error values depend on the specific solver.
    fn solve(&self, problem: &Problem<F>) -> Result<OptimizeResult<F>, LinearProgramError<F>>;
}

/// Outcome of a successful solve attempt.
pub struct OptimizeResult<F> {
    /// The solution vector
    x: Array1<F>,

    /// The cost function value
    fun: F,

    /// The number of iterations needed to find the solution
    iteration: usize,
}

impl<F> OptimizeResult<F> {
    pub(crate) fn new(x: Array1<F>, fun: F, iteration: usize) -> Self {
        Self { x, fun, iteration }
    }

    /// The number of iterations needed to find the solution
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// The cost function value
    pub fn fun(&self) -> &F {
        &self.fun
    }

    /// The solution vector
    pub fn x(&self) -> &Array1<F> {
        &self.x
    }
}
