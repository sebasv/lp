#![allow(non_snake_case)]
#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::Cholesky, cholesky::SolveCInplace, qr::QRDecomp, LinalgError};
#[cfg(feature = "blas")]
use ndarray_linalg::LeastSquaresSvdInto;
#[cfg(feature = "blas")]
use ndarray_linalg::SolveH;

use crate::error::LinearProgramError;
use linfa_linalg::qr::QR;

use crate::interior_point::linprog::Problem;

use ndarray::Axis;

use ndarray::Array1;

use ndarray::ArrayView1;

use crate::float::Float;

use ndarray::OwnedRepr;

use ndarray::Array2;

use super::rhat::Rhat;

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
pub enum EquationSolverType {
    Cholesky,
    Inverse,
    LeastSquares,
}
impl EquationSolverType {
    pub(crate) fn build<F: Float>(&self, A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        match self {
            EquationSolverType::Cholesky => EquationsSolver::cholesky(A),
            EquationSolverType::Inverse => EquationsSolver::inv(A),
            EquationSolverType::LeastSquares => EquationsSolver::least_squares(A),
        }
    }
}

pub(crate) enum EquationsSolver<F> {
    Cholesky(Array2<F>),
    Inv(QRDecomp<F, OwnedRepr<F>>),
    LstSq(QRDecomp<F, OwnedRepr<F>>, bool),
}

impl<F: Float> EquationsSolver<F> {
    pub(crate) fn cholesky(A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorizeh_into()?;
        #[cfg(not(feature = "blas"))]
        let factor = A.cholesky()?;
        Ok(EquationsSolver::Cholesky(factor))
    }
    pub(crate) fn inv(A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorize_into()?;
        #[cfg(not(feature = "blas"))]
        let factor = A.qr()?;
        Ok(EquationsSolver::Inv(factor)) // TODO can do with buffering, maybe QR
    }
    pub(crate) fn least_squares(A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorize_into()?;
        #[cfg(not(feature = "blas"))]
        let (factor, transposed) = if A.nrows() >= A.ncols() {
            (A.qr()?, false)
        } else {
            (A.view().reversed_axes().qr()?, true)
        };
        Ok(EquationsSolver::LstSq(factor, transposed)) // TODO can do with buffering, maybe QR
    }
    pub(crate) fn solve(&mut self, b: ArrayView1<F>) -> Result<Array1<F>, LinalgError> {
        let b2 = b.insert_axis(Axis(1)).into_owned();
        #[cfg(feature = "blas")]
        match self {
            EquationsSolver::Cholesky(A) | EquationsSolver::SymPos(A) => {
                A.solveh_into(b2).remove_axis(Axis(1))
            }
            EquationsSolver::Inv(A) => A.solve_into(b2),
            EquationsSolver::LstSq(A) => Ok(A.least_squares_into(b2)?.remove_axis(Axis(1))),
        }
        #[cfg(not(feature = "blas"))]
        let solved = match self {
            EquationsSolver::Cholesky(A) => A.solvec_into(b2)?,
            EquationsSolver::Inv(A) => A.solve_into(b2)?,
            EquationsSolver::LstSq(A, transposed) => {
                if *transposed {
                    A.solve_tr_into(b2)?
                } else {
                    // If array is fat (rows < cols) then take the QR of the transpose and run the
                    // transpose solving algorithm
                    A.solve_into(b2)?
                }
            }
        };
        Ok(solved.remove_axis(Axis(1)))
    }

    /// Attempt to solve a system of equations with the specified solver. If solving fails, retry with the next solver.
    /// If all solvers fail, a solver failed to be even created or we observe NaN in the output, fail with a NumericalProblem
    /// error.
    pub(crate) fn solve_round(
        mut self,
        Dinv: &Array1<F>,
        problem: &Problem<F>,
        x: &Array1<F>,
        rhat: &Rhat<F>,
        M: &Array2<F>,
    ) -> Result<
        (
            Array1<F>,
            Array1<F>,
            Array1<F>,
            Array1<F>,
            EquationsSolver<F>,
        ),
        LinearProgramError<F>,
    > {
        // [1] Equation 8.28
        // [1] Equation 8.29
        if let (Ok((p, q)), Ok((u, v))) = (
            self.sym_solve(Dinv, &problem.A, &problem.c, &problem.b),
            self.sym_solve(Dinv, &problem.A, &(&rhat.d - &(&rhat.xs / x)), &rhat.p),
        ) {
            if p.fold(false, |acc, e| acc || e.is_nan())
                || q.fold(false, |acc, e| acc || e.is_nan())
            {
                return Err(LinearProgramError::NumericalProblem);
            }
            return Ok((p, q, u, v, self));
        }
        // Solving failed due to numerical problems.
        // Usually this doesn't happen. If it does, it happens when
        // there are redundant constraints or when approaching the
        // solution. If so, change solver.
        match self {
            EquationsSolver::Cholesky(_) => EquationsSolver::inv(M)
                .or(Err(LinearProgramError::NumericalProblem))?
                .solve_round(Dinv, problem, x, rhat, M),
            EquationsSolver::Inv(_) => EquationsSolver::least_squares(M)
                .or(Err(LinearProgramError::NumericalProblem))?
                .solve_round(Dinv, problem, x, rhat, M),
            EquationsSolver::LstSq(..) => Err(LinearProgramError::NumericalProblem),
        }
    }

    #[inline]
    /// An implementation of [1] equation 8.31 and 8.32
    pub(crate) fn sym_solve(
        &mut self,
        Dinv: &Array1<F>,
        A: &Array2<F>,
        r1: &Array1<F>,
        r2: &Array1<F>,
    ) -> Result<(Array1<F>, Array1<F>), LinalgError> {
        let r = r2 + &A.dot(&(Dinv * r1));
        let v = self.solve(r.view())?;
        // [1] 8.32
        let u = Dinv * &(A.t().dot(&v) - r1);
        Ok((u, v))
    }
}
