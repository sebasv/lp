#![allow(non_snake_case)]
#[cfg(not(feature = "blas"))]
use crate::float::Lapack;
#[cfg(not(feature = "blas"))]
use linfa_linalg::{
    cholesky::Cholesky,
    cholesky::SolveCInplace,
    qr::{QRDecomp, QR},
    LinalgError,
};
#[cfg(feature = "blas")]
use ndarray_linalg::{
    error::LinalgError, FactorizeC, LeastSquaresSvdInto, QRSquare, SolveC, QR, UPLO,
};
use ndarray_linalg::{CholeskyFactorized, Factorize, LUFactorized, LeastSquaresSvd, Solve};

use ndarray::prelude::*;
use ndarray::OwnedRepr;

use crate::error::LinearProgramError;
use crate::float::Float;
use crate::linear_program::Problem;

use super::feasible_point::FeasiblePoint;
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
#[derive(PartialEq, Eq, Debug)]
pub enum EquationSolverType {
    Cholesky,
    Inverse,
    LeastSquares,
}
impl EquationSolverType {
    pub(crate) fn build<F: Float>(
        &self,
        point: &FeasiblePoint<F>,
        problem: &Problem<F>,
    ) -> Result<EquationsSolver<F>, LinearProgramError<F>> {
        //  Assemble M from [1] Equation 8.31
        let Dinv = &point.x / &point.z;
        let M = problem
            .A()
            .dot(&(&Dinv.clone().insert_axis(ndarray::Axis(1)) * &problem.A().t()));
        match self {
            EquationSolverType::Cholesky => EquationsSolver::cholesky(M, Dinv),
            EquationSolverType::Inverse => EquationsSolver::inv(M, Dinv),
            EquationSolverType::LeastSquares => EquationsSolver::least_squares(M, Dinv),
        }
        .or(Err(LinearProgramError::NumericalProblem))
    }
}

#[cfg(feature = "blas")]
pub(crate) enum EquationsSolver<F: Float> {
    Cholesky {
        factor: CholeskyFactorized<OwnedRepr<F>>,
        M: Array2<F>,
        Dinv: Array1<F>,
    },
    Inv {
        factor: LUFactorized<OwnedRepr<F>>,
        M: Array2<F>,
        Dinv: Array1<F>,
    },
    LstSq {
        M: Array2<F>,
        Dinv: Array1<F>,
    },
}

#[cfg(feature = "blas")]
impl<F: Float> EquationsSolver<F> {
    fn cholesky(M: Array2<F>, Dinv: Array1<F>) -> Result<EquationsSolver<F>, LinalgError> {
        let factor = M.factorizec(UPLO::Upper)?;
        Ok(EquationsSolver::Cholesky { factor, M, Dinv })
    }
    fn inv(M: Array2<F>, Dinv: Array1<F>) -> Result<EquationsSolver<F>, LinalgError> {
        let factor = M.factorize()?;
        Ok(EquationsSolver::Inv { factor, M, Dinv }) // TODO can do with buffering, maybe QR
    }
    fn least_squares(M: Array2<F>, Dinv: Array1<F>) -> Result<EquationsSolver<F>, LinalgError> {
        Ok(EquationsSolver::LstSq { M, Dinv }) // TODO can do with buffering, maybe QR
    }
    fn solve(&mut self, b: &Array1<F>) -> Result<Array1<F>, LinalgError> {
        match self {
            EquationsSolver::Cholesky { factor, .. } => factor.solvec(b),
            EquationsSolver::Inv { factor, .. } => factor.solve(b),
            EquationsSolver::LstSq { M, .. } => M.least_squares(b).map(|r| r.solution),
        }
    }
}

#[cfg(not(feature = "blas"))]
pub(crate) enum EquationsSolver<F> {
    Cholesky {
        factor: CholeskyFactorized<OwnedRepr<F>>,
        factor: Array2<F>,
        M: Array2<F>,
        Dinv: Array1<F>,
    },
    Inv {
        factor: QRDecomp<F, OwnedRepr<F>>,
        M: Array2<F>,
        Dinv: Array1<F>,
    },
    LstSq {
        factor: QRDecomp<F, OwnedRepr<F>>,
        transposed: bool,
        _M: Array2<F>,
        Dinv: Array1<F>,
    },
}

#[cfg(not(feature = "blas"))]
impl<F: Float> EquationsSolver<F> {
    fn cholesky(M: Array2<F>, Dinv: Array1<F>) -> Result<EquationsSolver<F>, LinalgError> {
        let factor = M.cholesky()?;
        Ok(EquationsSolver::Cholesky { factor, M, Dinv })
    }
    fn inv(M: Array2<F>, Dinv: Array1<F>) -> Result<EquationsSolver<F>, LinalgError> {
        let factor = M.qr()?;
        Ok(EquationsSolver::Inv { factor, M, Dinv }) // TODO can do with buffering, maybe QR
    }
    fn least_squares(M: Array2<F>, Dinv: Array1<F>) -> Result<EquationsSolver<F>, LinalgError> {
        let (factor, transposed) = if M.nrows() >= M.ncols() {
            (M.qr()?, false)
        } else {
            (M.view().reversed_axes().qr()?, true)
        };
        Ok(EquationsSolver::LstSq {
            factor,
            transposed,
            _M: M,
            Dinv,
        }) // TODO can do with buffering, maybe QR
    }

    fn solve(&mut self, b: &Array1<F>) -> Result<Array1<F>, LinalgError> {
        let b2 = b.view().insert_axis(Axis(1)).into_owned();
        let solved = match self {
            EquationsSolver::Cholesky { factor, .. } => factor.solvec_into(b2)?,
            EquationsSolver::Inv { factor, .. } => factor.solve_into(b2)?,
            EquationsSolver::LstSq {
                factor, transposed, ..
            } => {
                if *transposed {
                    factor.solve_tr_into(b2)?
                } else {
                    // If array is fat (rows < cols) then take the QR of the transpose and run the
                    // transpose solving algorithm
                    factor.solve_into(b2)?
                }
            }
        };
        Ok(solved.remove_axis(Axis(1)))
    }
}

impl<F: Float> EquationsSolver<F> {
    /// Attempt to solve a system of equations with the specified solver. If solving fails, retry with the next solver.
    /// If all solvers fail, a solver failed to be even created or we observe NaN in the output, fail with a NumericalProblem
    /// error.
    pub(crate) fn solve_newton_equations(
        mut self,
        // Dinv: &Array1<F>,
        problem: &Problem<F>,
        x: &Array1<F>,
        rhat: &Rhat<F>,
        // M: &Array2<F>,
    ) -> Result<(NewtonResult<F>, EquationsSolver<F>), LinearProgramError<F>> {
        // [1] Equation 8.28
        // [1] Equation 8.29
        if let (Ok((p, q)), Ok((u, v))) = (
            self.sym_solve(problem.A(), problem.c(), problem.b()),
            self.sym_solve(problem.A(), &(&rhat.d - &(&rhat.xs / x)), &rhat.p),
        ) {
            if p.fold(false, |acc, e| acc || e.is_nan())
                || q.fold(false, |acc, e| acc || e.is_nan())
            {
                return Err(LinearProgramError::NumericalProblem);
            }
            return Ok((NewtonResult { p, q, u, v }, self));
        }
        // Solving failed due to numerical problems.
        // Usually this doesn't happen. If it does, it happens when
        // there are redundant constraints or when approaching the
        // solution. If so, change solver.
        match self {
            EquationsSolver::Cholesky { M, Dinv, .. } => EquationsSolver::inv(M, Dinv)
                .or(Err(LinearProgramError::NumericalProblem))?
                .solve_newton_equations(problem, x, rhat),
            EquationsSolver::Inv { M, Dinv, .. } => EquationsSolver::least_squares(M, Dinv)
                .or(Err(LinearProgramError::NumericalProblem))?
                .solve_newton_equations(problem, x, rhat),
            EquationsSolver::LstSq { .. } => Err(LinearProgramError::NumericalProblem),
        }
    }

    #[inline]
    /// An implementation of [1] equation 8.31 and 8.32
    fn sym_solve(
        &mut self,
        A: &Array2<F>,
        r1: &Array1<F>,
        r2: &Array1<F>,
    ) -> Result<(Array1<F>, Array1<F>), LinalgError> {
        let r = r2 + &A.dot(&(self.Dinv() * r1));
        let v = self.solve(&r)?;
        // [1] 8.32
        let u = self.Dinv() * &(A.t().dot(&v) - r1);
        Ok((u, v))
    }

    #[inline]
    fn Dinv(&self) -> &Array1<F> {
        match self {
            EquationsSolver::Cholesky { Dinv, .. }
            | EquationsSolver::Inv { Dinv, .. }
            | EquationsSolver::LstSq { Dinv, .. } => Dinv,
        }
    }
}

pub(crate) struct NewtonResult<F> {
    pub(crate) p: Array1<F>,
    pub(crate) q: Array1<F>,
    pub(crate) u: Array1<F>,
    pub(crate) v: Array1<F>,
}
