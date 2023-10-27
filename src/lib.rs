//! A pure-Rust Interior Point solver for linear programs with equality and inequality constraints.
//!
//! # Linear programs
//!
//! A linear program is a mathematical optimization problem defined as:
//!
//! ```text
//!    min_x c'x
//!    st A_eq'x == b_eq
//!       A_ub'x <= b_ub
//!            x >= 0
//! ```
//!
//!
//!
//! # Example
//! ```
//! use approx::assert_abs_diff_eq;
//! use ndarray::array;
//!
//! use lp::Problem;
//! use lp::solvers::InteriorPoint;
//! use lp::solvers::interior_point::EquationSolverType;
//!
//!
//! let A_ub = array![[-3f64, 1.], [1., 2.]];
//! let b_ub = array![6., 4.];
//! let A_eq = array![[1., 1.]];
//! let b_eq = array![1.];
//! let c = array![-1., 4.];
//!
//! let problem = Problem::target(&c)
//!     // If you define neither equality nor inequality constraints,
//!     // the problem returns as unconstrained.
//!     .ub(&A_ub, &b_ub)
//!     .eq(&A_eq, &b_eq)
//!     .build()
//!     .unwrap();
//!
//!     // These are the default values you can overwrite.
//!     // You may omit any option for which the default is good enough for you
//! let solver = InteriorPoint::default()
//!     .solver_type(EquationSolverType::Cholesky)
//!     .tol(1e-8)
//!     .disp(false)
//!     .ip(true)
//!     .alpha0(0.99995)
//!     .max_iter(1000)
//!     .build()
//!     .unwrap();
//!
//! let res = solver.solve(&problem).unwrap();
//!
//! assert_abs_diff_eq!(res.x(), &array![1., 0.], epsilon = 1e-6);
//! ```
//!
//! # Feature flags
//!
//! ### `[blas]`
//! This package comes with the option to use BLAS-based solvers for systems of equations. To enable BLAS, set the `blas`
//! feature.

pub mod error;
pub(crate) mod float;
pub mod linear_program;
pub mod solvers;

// pub use interior_point::linprog::{EquationSolverType, InteriorPoint};
pub use linear_program::{OptimizeResult, Problem, ProblemBuilder};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use crate::solvers::InteriorPoint;
    use crate::Problem;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn make_problem() -> Problem<f64> {
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
        problem
    }
    #[test]
    fn test_problem_interface() {
        let problem = make_problem();
        problem.A();
        problem.b();
        problem.c();
        problem.c0();
    }

    #[test]
    fn test_interior_point_interface() {
        let problem = make_problem();
        let solver = InteriorPoint::builder().build().unwrap();
        let res = solver.solve(&problem).unwrap();

        assert_abs_diff_eq!(*res.x(), array![1., 0.], epsilon = 1e-6);
    }
}
