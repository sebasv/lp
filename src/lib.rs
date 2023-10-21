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
//! use lp::{LinearProgram, EquationSolverType};
//! use approx::assert_abs_diff_eq;
//! use ndarray::array;
//!
//!
//! let A_ub = array![[-3f64, 1.], [1., 2.]];
//! let b_ub = array![6., 4.];
//! let A_eq = array![[1., 1.]];
//! let b_eq = array![1.];
//! let c = array![-1., 4.];
//!
//! let res = LinearProgram::target(c.view())
//!     // If you define neither equality nor inequality constraints,
//!     // the problem returns as unconstrained.
//!     .ub(A_ub.view(), b_ub.view())
//!     .eq(A_eq.view(), b_eq.view())
//!     // These are the default values you can overwrite.
//!     // You may omit any option for which the default is good enough for you
//!     .solver_type(EquationSolverType::Cholesky)
//!     .tol(1e-8)
//!     .disp(false)
//!     .ip(true)
//!     .alpha0(0.99995)
//!     .max_iter(1000)
//!     .build()
//!     .unwrap()
//!     .solve()
//!     .unwrap();
//!
//! assert_abs_diff_eq!(res.x, array![1., 0.], epsilon = 1e-6);
//! ```
//!
//! # Feature flags
//!
//! ### `[blas]`
//! This package comes with the option to use BLAS-based solvers for systems of equations. To enable BLAS, set the `blas`
//! feature.

pub(crate) mod float;
pub(crate) mod linprog;

pub use linprog::{EquationSolverType, LinearProgram};
