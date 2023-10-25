#![allow(non_snake_case)]
//! Implementation of the MOSEK [1] interior point solver for linear programs.
//! This implementation is a variation of the [Mehrotra predictor-corrector method](https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method).
//!
//! This is the same algorithm as the interior point method implemented in the Python package SciPy,
//! and the SciPy implementation was used as a reference during development of this Rust implementation.
//! The author of this package is grateful to the SciPy maintainers for allowing such use of their source code.
//!
//! This attribution should explicitly not be interpreted as an endorsement from SciPy or its maintainers.
//!
//! .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
//!        optimizer for linear programming: an implementation of the
//!        homogeneous algorithm." High performance optimization. Springer US,
//!        2000. 197-232.
//!
use crate::error::LinearProgramError;
use crate::float::Float;
use ndarray::{concatenate, s, Array};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub use super::equations_solver::EquationSolverType;
use super::feasible_point::FeasiblePoint;

pub(crate) struct Indicators<F> {
    /// primal infeasibility
    pub(crate) rho_p: F,
    /// dual infeasibility
    pub(crate) rho_d: F,
    /// number of significant digits in objective value
    pub(crate) rho_A: F,
    /// gap infeasibility
    pub(crate) rho_g: F,

    pub(crate) rho_mu: F,
    /// Primal objective value
    pub(crate) obj: F,
    /// Dual objective value
    pub(crate) bty: F,
}
impl<F: Float> Indicators<F> {
    /// Indicators show that infeasibility gaps are almost closed -> it does not get any better.
    fn smaller_than(&self, tol: F) -> bool {
        self.rho_p < tol && self.rho_d < tol && self.rho_A < tol
    }

    /// Indicators show that it does not get any better
    fn max_reduction_infeasibility_le(&self, tol: F) -> bool {
        self.rho_p < tol && self.rho_d < tol && self.rho_g < tol
    }

    fn status(&self, tau: F, kappa: F, tol: F) -> Status {
        let tau_too_small = tau < tol * kappa.max(F::one());
        let inf1 = self.max_reduction_infeasibility_le(tol) && tau_too_small;
        let inf2 = self.rho_mu < tol && tau_too_small;
        if inf1 || inf2 {
            // [1] Lemma 8.4 / Theorem 8.3
            if self.bty > tol {
                Status::Infeasible
            } else {
                Status::Unbounded
            }
        } else if self.smaller_than(tol) {
            // [1] Statement after Theorem 8.2
            Status::Optimal
        } else {
            Status::Unfinished
        }
    }
}
enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    Unfinished,
}

pub struct Problem<F> {
    pub A: Array2<F>,
    pub b: Array1<F>,
    pub c: Array1<F>,
    pub c0: F,
}

impl<F: Float> Problem<F> {
    /// Convert a problem to slack form.
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
    fn lp_problem_to_slack_form(
        c: ArrayView1<F>,
        ub: Option<(ArrayView2<F>, ArrayView1<F>)>,
        eq: Option<(ArrayView2<F>, ArrayView1<F>)>,
    ) -> Result<Problem<F>, LinearProgramError<F>> {
        let n = c.len();
        let (A_ub, b_ub) = ub
            .map(|(A, b)| (A.to_owned(), b.to_owned()))
            .unwrap_or((Array2::zeros((0, n)), Array1::zeros(0)));
        let (A_eq, b_eq) = eq
            .map(|(A, b)| (A.to_owned(), b.to_owned()))
            .unwrap_or((Array2::zeros((0, n)), Array1::zeros(0)));

        let (nrows_ub, ncols_ub) = A_ub.dim();
        let (nrows_eq, ncols_eq) = A_eq.dim();
        if nrows_ub + nrows_eq == 0 {
            return Err(LinearProgramError::Unconstrained);
        }
        if ncols_ub != ncols_eq
            || ncols_eq != c.len()
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
        let c = concatenate(Axis(0), &[c.view(), Array1::zeros(nrows_ub).view()])
            .or(Err(LinearProgramError::IncompatibleInputDimensions))?;
        Ok(Problem {
            A,
            b,
            c,
            c0: F::zero(),
        })
    }
}

/// print progress of convergence criteria to stdout
fn display_iter<F: Float>(indicators: &Indicators<F>, alpha: F, display_header: bool) {
    if display_header {
        println!(" rho_p     \t rho_d     \t rho_g     \t alpha     \t rho_mu    \t obj       ");
    }
    println!(
        "{:3.8}\\t{:3.8}\\t{:3.8}\\t{alpha:3.8}\\t{:3.8}\\t{:8.3}",
        indicators.rho_p, indicators.rho_d, indicators.rho_g, indicators.rho_mu, indicators.obj
    );
}

pub struct LinearProgramInput<'a, F> {
    c: ArrayView1<'a, F>,
    ub: Option<(ArrayView2<'a, F>, ArrayView1<'a, F>)>,
    eq: Option<(ArrayView2<'a, F>, ArrayView1<'a, F>)>,
    tol: F,
    disp: bool,
    ip: bool,
    solver_type: EquationSolverType,
    alpha0: F,
    max_iter: usize,
}

impl<'a, F: Float> LinearProgramInput<'a, F> {
    pub fn new(c: ArrayView1<'a, F>) -> LinearProgramInput<'a, F> {
        LinearProgramInput {
            c,
            ub: None,
            eq: None,
            tol: F::cast(1e-8),
            disp: false,
            ip: true,
            solver_type: EquationSolverType::Cholesky,
            alpha0: F::cast(0.99995),
            max_iter: 1000,
        }
    }

    /// Set an upper bound for the problem, such that `A @ x <= b`.
    /// Either an upper bound (`ub`), a lower bound (`eq`) or both should be specified.
    /// To prevent numerical problems, it is advisable to remove redundant constraints and to scale all constraints to
    /// roughly the same order of magnitude.
    pub fn ub(mut self, A: ArrayView2<'a, F>, b: ArrayView1<'a, F>) -> Self {
        self.ub = Some((A, b));
        self
    }

    /// Set an equality constraint for the problem, such that `A @ x == b`.
    /// Either an upper bound (`ub`), a lower bound (`eq`) or both should be specified.
    /// To prevent numerical problems, it is advisable to remove redundant constraints and to scale all constraints to
    /// roughly the same order of magnitude.
    pub fn eq(mut self, A: ArrayView2<'a, F>, b: ArrayView1<'a, F>) -> Self {
        self.eq = Some((A, b));
        self
    }

    /// Set the convergence tolerance. If all the convergence indicators are smaller than `tol`, optimization is
    /// successfully terminated. Should be a small positive value.
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set to true to print the value of convergence indicators to stdout at every iteration.
    pub fn disp(mut self, disp: bool) -> Self {
        self.disp = disp;
        self
    }

    /// Indicator whether to use an alternative initial point to seed the solver.
    /// There is no clear evidence that setting this is better or worse.
    pub fn ip(mut self, ip: bool) -> Self {
        self.ip = ip;
        self
    }

    /// Indicator which equation solver to try and use first.
    ///
    /// During each iteration, a system of linear equations must be solved multiple times.
    /// The fastest option is to create a Cholesky decomposition of the system and reuse this decomposition,
    /// but this decomposition may run into numerical difficulties near the solution of some problems.
    ///
    /// If this happens, the algorithm falls back to first a regular inverse, and then a least squares solution.
    ///
    /// Every fallback is expensive because a new decomposition must be calculated. If your problem is prone to frequently
    /// falling back to more robust equation solvers, you can set the initial equation solver to a more robust choice
    /// such that less fallbacks happen.
    pub fn solver_type(mut self, solver_type: EquationSolverType) -> Self {
        self.solver_type = solver_type;
        self
    }

    /// Step size multiplier.
    ///
    /// At each iteration the parameters are improved with a step in an informed search direction. This step is guaranteed
    /// to keep the solution feasible, but since this is an _interior_ point method we must make sure that a step does not
    /// return a point on the boundary of the feasible set, since then we won't be able to compute a next search direction.
    /// To prevent this, each step is multiplied by 0 < `alpha0` < 1. Smaller `alpha0` values result in more iterations
    /// but potentially a more stable search path.
    pub fn alpha0(mut self, alpha0: F) -> Self {
        self.alpha0 = alpha0;
        self
    }

    /// Maximum number of iterations before we give up on trying to solve the problem.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Construct a linear program from the provided inputs, validating the input values.
    /// Returns an `InvalidParameter` error if one of the input constraints is violated.
    pub fn build(self) -> Result<LinearProgram<F>, LinearProgramError<F>> {
        let problem = Problem::lp_problem_to_slack_form(self.c, self.ub, self.eq)?;
        if self.alpha0 <= F::zero() || self.alpha0 >= F::one() {
            return Err(LinearProgramError::InvalidParameter);
        }
        if self.tol <= F::zero() {
            return Err(LinearProgramError::InvalidParameter);
        }
        let n_slack = problem.c.len() - self.c.len();
        Ok(LinearProgram {
            problem,
            tol: self.tol,
            disp: self.disp,
            ip: self.ip,
            solver_type: self.solver_type,
            alpha0: self.alpha0,
            max_iter: self.max_iter,
            n_slack,
        })
    }
}

pub struct LinearProgram<F> {
    problem: Problem<F>,
    tol: F,
    disp: bool,
    ip: bool,
    solver_type: EquationSolverType,
    alpha0: F,
    max_iter: usize,
    n_slack: usize,
}

impl<F: Float> LinearProgram<F> {
    /// Construct a new linear program, to be customized through the builder pattern.
    /// Takes as an argument the cost function vector.
    ///
    /// ```rust
    /// use lp::LinearProgram;
    /// use ndarray::array;
    ///
    ///
    /// let A_ub = array![[-3f64, 1.], [1., 2.]];
    /// let b_ub = array![6., 4.];
    /// let c = array![-1., 4.];
    ///
    /// let res = LinearProgram::target(c.view())
    ///     .ub(A_ub.view(), b_ub.view())
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn target(c: ArrayView1<F>) -> LinearProgramInput<F> {
        LinearProgramInput::new(c)
    }

    /// Attempt to solve the linear program.
    pub fn solve(&self) -> Result<OptimizeResult<F>, LinearProgramError<F>> {
        let (x_slack, iteration) = self.ip_hsd()?;
        // Eliminate artificial variables, re-introduce presolved variables, etc.
        let x = x_slack.slice(s![..x_slack.len() - self.n_slack]).to_owned();
        let fun = self.problem.c.dot(&x_slack) + self.problem.c0;
        Ok(OptimizeResult { x, fun, iteration })
    }

    /// Solve a linear programming problem in standard form:
    ///
    /// Minimize:
    ///```text
    ///     c @ x
    ///```
    /// Subject to:
    ///```text
    ///     A @ x == b
    ///         x >= 0
    ///```
    /// using the interior point method of [1].
    fn ip_hsd(&self) -> Result<(Array1<F>, usize), LinearProgramError<F>> {
        let mut point = FeasiblePoint::blind_start(&self.problem);

        // [1] 4.5
        let indicators = point.indicators(&self.problem);

        if self.disp {
            display_iter(&indicators, F::one(), true);
        }
        let mut ip = self.ip;
        for iteration in 0..self.max_iter {
            // Solve [1] 8.6 and 8.7/8.13/8.23
            let delta = point.get_delta(&self.problem, &self.solver_type, ip)?;
            let alpha = if ip {
                F::one()
            } else {
                // [1] Section 4.3
                point.get_step_size(&delta, self.alpha0)
            };
            point = point.do_step(&delta, alpha, ip);
            ip = false;

            let indicators = point.indicators(&self.problem);

            if self.disp {
                display_iter(&indicators, alpha, false);
            }
            match indicators.status(point.tau, point.kappa, self.tol) {
                Status::Optimal => return Ok((&point.x / point.tau, iteration)),
                Status::Infeasible => return Err(LinearProgramError::Infeasible),
                Status::Unbounded => return Err(LinearProgramError::Unbounded),
                Status::Unfinished => {}
            };
        }
        Err(LinearProgramError::IterationLimitExceeded(
            point.x / point.tau,
        ))
    }
}

/// Outcome of a successful solve attempt.
pub struct OptimizeResult<F> {
    /// The solution vector
    pub x: Array1<F>,

    /// The cost function value
    pub fun: F,

    // The number of iterations needed to find the solution
    pub iteration: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_linprog_builder() {
        let A_ub = array![[-3f64, 1.], [1., 2.]];
        let b_ub = array![6., 4.];
        let A_eq = array![[1., 1.]];
        let b_eq = array![1.];
        let c = array![-1., 4.];

        let res = LinearProgram::target(c.view())
            .ub(A_ub.view(), b_ub.view())
            .eq(A_eq.view(), b_eq.view())
            .build()
            .unwrap()
            .solve()
            .unwrap();

        assert_abs_diff_eq!(res.x, array![1., 0.], epsilon = 1e-6);
    }
    #[test]
    fn test_linprog_eq_only() {
        let A_eq = array![[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0],];
        let b_eq = array![1.0, 2.0, 3.0];
        let c = array![-1.0, 4.0, -1.2];

        let res = LinearProgram::target(c.view())
            .eq(A_eq.view(), b_eq.view())
            .build()
            .unwrap()
            .solve()
            .unwrap();

        assert_abs_diff_eq!(res.x, array![1. / 3., 1. / 3., 4. / 3.], epsilon = 1e-6);
    }
    #[test]
    fn test_linprog_ub_only() {
        let A_ub = array![[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0],];
        let b_ub = array![1.0, 2.0, 3.0];
        let c = array![-1.0, 4.0, -1.2];

        let res = LinearProgram::target(c.view())
            .ub(A_ub.view(), b_ub.view())
            .build()
            .unwrap()
            .solve()
            .unwrap();

        assert_abs_diff_eq!(res.x, array![0.5, 0.0, 1.25], epsilon = 1e-6);
    }
}
