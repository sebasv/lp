#![allow(non_snake_case)]
use crate::{error::LinearProgramError, float::Float};
use ndarray::{concatenate, prelude::*};
pub struct Problem<F> {
    A: Array2<F>,
    b: Array1<F>,
    c: Array1<F>,
    c0: F,
    n_slack: usize,
}

impl<F: Float> Problem<F> {
    pub fn target(c: &Array1<F>) -> ProblemBuilder<F> {
        ProblemBuilder::new(c)
    }

    pub fn A(&self) -> &Array2<F> {
        &self.A
    }

    pub fn b(&self) -> &Array1<F> {
        &self.b
    }

    pub fn c(&self) -> &Array1<F> {
        &self.c
    }

    pub fn c0(&self) -> F {
        self.c0
    }

    pub fn n_slack(&self) -> usize {
        self.n_slack
    }

    pub(crate) fn denormalize_target(&self, x_slack: &Array1<F>) -> F {
        self.c.dot(x_slack) + self.c0
    }

    pub(crate) fn denormalize_x_into(&self, x_slack: Array1<F>) -> Array1<F> {
        x_slack
            .slice(s![..x_slack.len() - self.n_slack])
            .into_owned()
    }
}

/// Outcome of a successful solve attempt.
pub struct OptimizeResult<F> {
    /// The solution vector
    x: Array1<F>,

    /// The cost function value
    fun: F,

    // The number of iterations needed to find the solution
    iteration: usize,
}

impl<F> OptimizeResult<F> {
    pub fn new(x: Array1<F>, fun: F, iteration: usize) -> Self {
        Self { x, fun, iteration }
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn fun(&self) -> &F {
        &self.fun
    }

    pub fn x(&self) -> &Array1<F> {
        &self.x
    }
}

pub struct ProblemBuilder<'a, F> {
    c: &'a Array1<F>,
    ub: Option<(&'a Array2<F>, &'a Array1<F>)>,
    eq: Option<(&'a Array2<F>, &'a Array1<F>)>,
}

impl<'a, F: Float> ProblemBuilder<'a, F> {
    pub fn new(c: &'a Array1<F>) -> ProblemBuilder<'a, F> {
        ProblemBuilder {
            c,
            ub: None,
            eq: None,
        }
    }

    /// Set an upper bound for the problem, such that `A @ x <= b`.
    /// Either an upper bound (`ub`), a lower bound (`eq`) or both should be specified.
    /// To prevent numerical problems, it is advisable to remove redundant constraints and to scale all constraints to
    /// roughly the same order of magnitude.
    pub fn ub(mut self, A: &'a Array2<F>, b: &'a Array1<F>) -> Self {
        self.ub = Some((A, b));
        self
    }

    /// Set an equality constraint for the problem, such that `A @ x == b`.
    /// Either an upper bound (`ub`), a lower bound (`eq`) or both should be specified.
    /// To prevent numerical problems, it is advisable to remove redundant constraints and to scale all constraints to
    /// roughly the same order of magnitude.
    pub fn eq(mut self, A: &'a Array2<F>, b: &'a Array1<F>) -> Self {
        self.eq = Some((A, b));
        self
    }

    /// Construct a linear program from the provided inputs, validating the input values.
    /// Converts the problem to standard form.
    ///
    /// If
    /// min_x c'x
    ///    st A_eq'x == b_eq,
    ///       A_ub'x <= b_ub,
    ///            x >= 0
    ///
    /// then
    /// min_{x,s} c'x + 0's
    ///        st [ A_ub I ]'[x] == [b_ub]
    ///           [ A_eq O ] [s]    [b_eq]
    ///                      x,s >= 0
    ///
    /// is the analogous problem with only equality constraints.
    ///
    /// Returns an error if any of the dimensions do not conform to the definition above, or if there are no constraints.
    pub fn build(self) -> Result<Problem<F>, LinearProgramError<F>> {
        let n = self.c.len();
        let A_empty = Array2::zeros((0, n));
        let b_empty = Array1::zeros(0);
        let (A_ub, b_ub) = self.ub.map(|(A, b)| (A, b)).unwrap_or((&A_empty, &b_empty));
        let (A_eq, b_eq) = self.eq.map(|(A, b)| (A, b)).unwrap_or((&A_empty, &b_empty));

        let (nrows_ub, ncols_ub) = A_ub.dim();
        let (nrows_eq, ncols_eq) = A_eq.dim();
        if nrows_ub + nrows_eq == 0 {
            return Err(LinearProgramError::Unconstrained);
        }
        if ncols_ub != ncols_eq
            || ncols_eq != self.c.len()
            || nrows_ub != b_ub.len()
            || nrows_eq != b_eq.len()
        {
            return Err(LinearProgramError::IncompatibleInputDimensions);
        }

        let A1 = concatenate(Axis(0), &[A_ub.view(), A_eq.view()])
            .or(Err(LinearProgramError::IncompatibleInputDimensions))?;
        let A2 = concatenate(
            Axis(0),
            &[
                Array::eye(nrows_ub).view(),
                Array2::zeros((nrows_eq, nrows_ub)).view(),
            ],
        )
        .or(Err(LinearProgramError::IncompatibleInputDimensions))?;
        let A = concatenate(Axis(1), &[A1.view(), A2.view()])
            .or(Err(LinearProgramError::IncompatibleInputDimensions))?;
        let b = concatenate(Axis(0), &[b_ub.view(), b_eq.view()])
            .or(Err(LinearProgramError::IncompatibleInputDimensions))?;
        let c_slack = concatenate(Axis(0), &[self.c.view(), Array1::zeros(nrows_ub).view()])
            .or(Err(LinearProgramError::IncompatibleInputDimensions))?;
        let n_slack = c_slack.len() - self.c.len();
        Ok(Problem {
            A,
            b,
            c: c_slack,
            c0: F::zero(),
            n_slack,
        })
    }
}
