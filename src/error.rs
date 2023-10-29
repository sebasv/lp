//! Error type used by solvers and pre-solvers.
use ndarray::Array1;
use std::fmt::Debug;
use thiserror::Error;
#[derive(Error, Debug)]
/// Error variant for problems encountered while attempting to solve a linear program.
pub enum LinearProgramError<F: Debug> {
    #[error("The problem is unconstrained, meaning the solution is the all-zeros vector if `c` is nonnegative, or unbounded otherwise.")]
    #[doc = "The problem is unconstrained, meaning the solution is the all-zeros vector if `c` is nonnegative, or unbounded otherwise."]
    Unconstrained,
    #[error("The solver encountered numerical problems it could not recover from. Likely causes are linearly dependent constraints or variables whose scale differs by multiple orders of magnitude.")]
    #[doc = "The solver encountered numerical problems it could not recover from. Likely causes are linearly dependent constraints or variables whose scale differs by multiple orders of magnitude."]
    NumericalProblem,
    #[error("A parameter was set to an invalid value: {0}")]
    #[doc = "A parameter was set to an invalid value: {0}"]
    InvalidParameter(&'static str),
    #[error("The dimensions of your cost- and constraint arrays do not align.")]
    #[doc = "The dimensions of your cost- and constraint arrays do not align."]
    IncompatibleInputDimensions,
    #[error("The solver finished successfully, it appears that the problem is infeasible.")]
    #[doc = "The solver finished successfully, it appears that the problem is infeasible."]
    Infeasible,
    #[error("The solver finished successfully, it appears that your problem is unbounded.")]
    #[doc = "The solver finished successfully, it appears that your problem is unbounded."]
    Unbounded,
    #[error("The solver failed to converge within the maximum number of iterations. Best solution after the final iteration:\n{0:#?}")]
    #[doc = "The solver failed to converge within the maximum number of iterations. Best solution after the final iteration:\n{0:#?}"]
    IterationLimitExceeded(Array1<F>),
}
