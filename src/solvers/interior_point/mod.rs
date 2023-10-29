#![allow(non_snake_case)]
//! Implementation of the MOSEK \[1\] interior point solver for linear programs.
//! This implementation is a variation of the [Mehrotra predictor-corrector method](https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method).
//!
//! This is the same algorithm as the interior point method implemented in the Python package SciPy,
//! and the SciPy implementation was used as a reference during development of this Rust implementation.
//! The author of this package is grateful to the SciPy maintainers for allowing such use of their source code.
//!
//! This attribution should explicitly not be interpreted as an endorsement from SciPy or its maintainers.
//!
//! .. \[1\] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
//!        optimizer for linear programming: an implementation of the
//!        homogeneous algorithm." High performance optimization. Springer US,
//!        2000. 197-232.
//!
mod delta;
mod feasible_point;
mod indicators;
mod newton_equations;
mod residual;
mod rhat;

use crate::error::LinearProgramError;
use crate::float::Float;
use ndarray::Array1;

use crate::linear_program::Problem;
use feasible_point::FeasiblePoint;
use indicators::{Indicators, Status};
pub use newton_equations::EquationSolverType;

use crate::solvers::Solver;

use super::OptimizeResult;

/// Builder struct to customize the [`InteriorPoint`] solver.
///
/// After constructing the default solver with [`InteriorPointBuilder::new]`,
/// use the other methods to update specific settings, and finally call [`build`](InteriorPointBuilder::build) to validate
/// the customized settings and create the solver.
pub struct InteriorPointBuilder<F> {
    tol: F,
    disp: bool,
    ip: bool,
    solver_type: EquationSolverType,
    alpha0: F,
    max_iter: usize,
}

impl<F: Float> InteriorPointBuilder<F> {
    pub(crate) fn new() -> InteriorPointBuilder<F> {
        InteriorPointBuilder {
            tol: F::cast(1e-8),
            disp: false,
            ip: true,
            solver_type: EquationSolverType::Cholesky,
            alpha0: F::cast(0.99995),
            max_iter: 1000,
        }
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
    pub fn build(self) -> Result<InteriorPoint<F>, LinearProgramError<F>> {
        if self.alpha0 <= F::zero() || self.alpha0 >= F::one() {
            return Err(LinearProgramError::InvalidParameter(
                "Alpha0 must be between 0 and 1 (exclusive)",
            ));
        }
        if self.tol <= F::zero() {
            return Err(LinearProgramError::InvalidParameter(
                "The tolerance must be nonnegative.",
            ));
        }
        Ok(InteriorPoint {
            tol: self.tol,
            disp: self.disp,
            ip: self.ip,
            solver_type: self.solver_type,
            alpha0: self.alpha0,
            max_iter: self.max_iter,
        })
    }
}

#[derive(PartialEq, Eq, Debug)]
/// Interior point struct that can be used to solve linear programs.
///
/// To get started quickly, use the [`default`](InteriorPoint::default()) method to initialize the solver with default parameters.
/// See the [`custom`](InteriorPoint::custom()) for customization options through the builder pattern.
pub struct InteriorPoint<F> {
    tol: F,
    disp: bool,
    ip: bool,
    solver_type: EquationSolverType,
    alpha0: F,
    max_iter: usize,
}

impl<F: Float> Default for InteriorPoint<F> {
    /// The interior point solver with default configuration.
    fn default() -> Self {
        InteriorPointBuilder::new().build().unwrap()
    }
}

impl<F: Float> Solver<F> for InteriorPoint<F> {
    fn solve(&self, problem: &Problem<F>) -> Result<OptimizeResult<F>, LinearProgramError<F>> {
        let (x_slack, iteration) = self.solve_normal_form(problem)?;
        // Eliminate artificial variables, re-introduce presolved variables, etc.
        let fun = problem.denormalize_target(&x_slack);
        let x = problem.denormalize_x_into(x_slack);
        Ok(OptimizeResult::new(x, fun, iteration))
    }
}

impl<F: Float> InteriorPoint<F> {
    /// Construct a new linear program, to be customized through the builder pattern.
    /// Takes as an argument the cost function vector.
    ///
    /// ```rust
    /// use approx::assert_abs_diff_eq;
    /// use lp::prelude::*;
    /// use ndarray::array;
    ///
    ///
    /// let A_ub = array![[-3f64, 1.], [1., 2.]];
    /// let b_ub = array![6., 4.];
    /// let c = array![-1., 4.];
    ///
    /// let problem = Problem::target(&c)
    ///     .ub(&A_ub, &b_ub)
    ///     .build()
    ///     .unwrap();
    /// let solver = InteriorPoint::custom().build().unwrap();
    /// let res = solver.solve(&problem).unwrap();
    ///
    /// assert_abs_diff_eq!(*res.x(), array![4., 0.], epsilon = 1e-6);
    ///
    /// ```
    pub fn custom() -> InteriorPointBuilder<F> {
        InteriorPointBuilder::new()
    }

    fn solve_normal_form(
        &self,
        problem: &Problem<F>,
    ) -> Result<(Array1<F>, usize), LinearProgramError<F>> {
        let mut point = FeasiblePoint::blind_start(problem);

        // [1] 4.5
        let indicators = Indicators::from_point_and_problem(&point, problem);

        if self.disp {
            println!("alpha     \trho_p     \trho_d     \trho_g     \trho_mu    \tobj       ");
            println!("1.00000000\t{indicators}");
        }
        let mut ip = self.ip;
        for iteration in 1..=self.max_iter {
            // Solve [1] 8.6 and 8.7/8.13/8.23
            let delta = point.get_delta(problem, &self.solver_type, ip)?;
            let alpha = if ip {
                F::one()
            } else {
                // [1] Section 4.3
                point.get_step_size(&delta, self.alpha0)
            };
            point = point.do_step(&delta, alpha, ip);
            ip = false;

            let indicators = Indicators::from_point_and_problem(&point, problem);

            if self.disp {
                println!("{alpha:3.8}\t{indicators}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn default_builder_doesnt_panic() {
        let ip = InteriorPoint::<f64>::default();
        let ip_long_way_round = InteriorPoint::custom().build().unwrap();
        assert_eq!(ip, ip_long_way_round);
    }

    #[test]
    fn test_interior_point_builder() {
        let A_ub = array![[-3f64, 1.], [1., 2.]];
        let b_ub = array![6., 4.];
        let A_eq = array![[1., 1.]];
        let b_eq = array![1.];
        let c = array![-1., 4.];
        let problem = Problem::target(&c)
            .ub(&A_ub, &b_ub)
            .eq(&A_eq, &b_eq)
            .build()
            .unwrap();

        let solver = InteriorPoint::default();
        let res = solver.solve(&problem).unwrap();

        assert_abs_diff_eq!(*res.x(), array![1., 0.], epsilon = 1e-6);
    }

    #[test]
    fn test_interior_point_inverse_solver() {
        let A_ub = array![[-3f64, 1.], [1., 2.]];
        let b_ub = array![6., 4.];
        let A_eq = array![[1., 1.]];
        let b_eq = array![1.];
        let c = array![-1., 4.];
        let problem = Problem::target(&c)
            .ub(&A_ub, &b_ub)
            .eq(&A_eq, &b_eq)
            .build()
            .unwrap();

        let solver = InteriorPoint::custom()
            .solver_type(EquationSolverType::Inverse)
            .build()
            .unwrap();
        let res = solver.solve(&problem).unwrap();

        assert_abs_diff_eq!(*res.x(), array![1., 0.], epsilon = 1e-6);
    }

    #[test]
    fn test_interior_point_least_squares_solver() {
        let A_ub = array![[-3f64, 1.], [1., 2.]];
        let b_ub = array![6., 4.];
        let A_eq = array![[1., 1.]];
        let b_eq = array![1.];
        let c = array![-1., 4.];
        let problem = Problem::target(&c)
            .ub(&A_ub, &b_ub)
            .eq(&A_eq, &b_eq)
            .build()
            .unwrap();

        let solver = InteriorPoint::custom()
            .solver_type(EquationSolverType::LeastSquares)
            .build()
            .unwrap();
        let res = solver.solve(&problem).unwrap();

        assert_abs_diff_eq!(*res.x(), array![1., 0.], epsilon = 1e-6);
    }

    #[test]
    fn test_linprog_eq_only() {
        let A_eq = array![[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0],];
        let b_eq = array![1.0, 2.0, 3.0];
        let c = array![-1.0, 4.0, -1.2];

        let problem = Problem::target(&c).eq(&A_eq, &b_eq).build().unwrap();

        let solver = InteriorPoint::default();
        let res = solver.solve(&problem).unwrap();

        assert_abs_diff_eq!(*res.x(), array![1. / 3., 1. / 3., 4. / 3.], epsilon = 1e-6);
    }
    #[test]
    fn test_linprog_ub_only() {
        let A_ub = array![[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0],];
        let b_ub = array![1.0, 2.0, 3.0];
        let c = array![-1.0, 4.0, -1.2];

        let problem = Problem::target(&c).ub(&A_ub, &b_ub).build().unwrap();

        let solver = InteriorPoint::default();
        let res = solver.solve(&problem).unwrap();

        assert_abs_diff_eq!(*res.x(), array![0.5, 0.0, 1.25], epsilon = 1e-6);
    }
}
