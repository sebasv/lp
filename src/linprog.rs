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
use crate::float::Float;
use linfa_linalg::qr::QR;
use ndarray::{concatenate, s, Array, OwnedRepr};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use std::fmt::Debug;
use thiserror::Error;

#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::Cholesky, cholesky::SolveCInplace, qr::QRDecomp, LinalgError};
#[cfg(feature = "blas")]
use ndarray_linalg::LeastSquaresSvdInto;
#[cfg(feature = "blas")]
use ndarray_linalg::SolveH;

#[inline]
/// An implementation of [1] equation 8.31 and 8.32
fn sym_solve<F: Float>(
    Dinv: &Array1<F>,
    A: &Array2<F>,
    r1: &Array1<F>,
    r2: &Array1<F>,
    solver: &mut EquationsSolver<F>,
) -> Result<(Array1<F>, Array1<F>), LinalgError> {
    let r = r2 + &A.dot(&(Dinv * r1));
    let v = solver.solve(r.view())?;
    // [1] 8.32
    let u = Dinv * &(A.t().dot(&v) - r1);
    Ok((u, v))
}

struct Residuals<F> {
    rho_p: F,
    rho_d: F,
    rho_g: F,
    rho_mu: F,
}

impl<F: Float> Residuals<F> {
    fn calculate(
        problem: &Problem<F>,
        x: &Array1<F>,
        y: &Array1<F>,
        z: &Array1<F>,
        tau: F,
        kappa: F,
    ) -> Residuals<F> {
        // See [1], Section 4 - The Homogeneous Algorithm, Equation 8.8
        let norm = |a: &Array1<F>| a.dot(a).sqrt();
        let r_p = |x: &Array1<F>, tau: F| norm(&(&(&problem.b * tau) - &problem.A.dot(x)));
        let r_d = |y: &Array1<F>, z: &Array1<F>, tau: F| {
            norm(&(&(&problem.c * tau) - &problem.A.t().dot(y) - z))
        };
        let r_g =
            |x: &Array1<F>, y: &Array1<F>, kappa: F| kappa + problem.c.dot(x) - problem.b.dot(y);
        let mu = |x: &Array1<F>, tau: F, z: &Array1<F>, kappa: F| {
            (x.dot(z) + tau * kappa) / F::cast(x.len() + 1)
        };

        let rho_p = r_p(x, tau);
        let rho_d = r_d(y, z, tau);
        let rho_g = r_g(x, y, kappa).abs();
        let rho_mu = mu(x, tau, z, kappa);
        Residuals {
            rho_p,
            rho_d,
            rho_g,
            rho_mu,
        }
    }
}

struct Delta<F> {
    d_x: Array1<F>,
    d_y: Array1<F>,
    d_z: Array1<F>,
    d_tau: F,
    d_kappa: F,
}

impl<F: Float> Delta<F> {
    fn compute(
        point: &FeasiblePoint<F>,
        rhat: &Rhat<F>,
        problem: &Problem<F>,
        solver: EquationsSolver<F>,
        Dinv: &Array1<F>,
        M: &Array2<F>,
    ) -> Result<(Delta<F>, EquationsSolver<F>), LinearProgramError<F>> {
        let (p, q, u, v, new_solver) = solve_round(Dinv, problem, solver, &point.x, rhat, M)?;
        // [1] Results after 8.29
        let d_tau = (rhat.g + F::one() / point.tau * rhat.tk
            - (-problem.c.dot(&u) + problem.b.dot(&v)))
            / (F::one() / point.tau * point.kappa + (-problem.c.dot(&p) + problem.b.dot(&q)));
        let d_x = &u + &(&p * d_tau);
        let d_y = v + q * d_tau;

        // [1] Relations between  after 8.25 and 8.26
        let d_z = (&rhat.xs - &point.z * &d_x) / &point.x;
        let d_kappa = F::one() / point.tau * (rhat.tk - point.kappa * d_tau);
        Ok((
            Delta {
                d_x,
                d_y,
                d_z,
                d_tau,
                d_kappa,
            },
            new_solver,
        ))
    }
}

struct Indicators<F> {
    /// primal infeasibility
    rho_p: F,
    /// dual infeasibility
    rho_d: F,
    /// number of significant digits in objective value
    rho_A: F,
    /// gap infeasibility
    rho_g: F,

    rho_mu: F,
    /// Primal objective value
    obj: F,
    /// Dual objective value
    bty: F,
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

struct FeasiblePoint<F> {
    x: Array1<F>,
    y: Array1<F>,
    z: Array1<F>,
    tau: F,
    kappa: F,
    initial_residuals: Residuals<F>,
}

impl<F: Float> FeasiblePoint<F> {
    fn blind_start(problem: &Problem<F>) -> FeasiblePoint<F> {
        let (m, n) = problem.A.dim();
        let x = Array1::ones(n);
        let y = Array1::zeros(m);
        let z = Array1::ones(n);
        let tau = F::one();
        let kappa = F::one();
        FeasiblePoint {
            initial_residuals: Residuals::calculate(problem, &x, &y, &z, tau, kappa),
            x,
            y,
            z,
            tau,
            kappa,
        }
    }
    fn residuals(&self, problem: &Problem<F>) -> Residuals<F> {
        Residuals::calculate(problem, &self.x, &self.y, &self.z, self.tau, self.kappa)
    }

    /// [1] 4.5
    fn indicators(&self, problem: &Problem<F>) -> Indicators<F> {
        let obj = problem.c.dot(&(&self.x / self.tau)) + problem.c0;
        let bty = problem.b.t().dot(&self.y);
        let rho_A = (problem.c.t().dot(&self.x) - bty).abs()
            / (self.tau + problem.b.t().dot(&self.y).abs());
        let residuals = self.residuals(problem);
        Indicators {
            rho_p: residuals.rho_p / self.initial_residuals.rho_p.max(F::one()),
            rho_d: residuals.rho_d / self.initial_residuals.rho_d.max(F::one()),
            rho_A,
            rho_g: residuals.rho_g / self.initial_residuals.rho_g.max(F::one()),
            rho_mu: residuals.rho_mu / self.initial_residuals.rho_mu,
            obj,
            bty,
        }
    }

    #[inline]
    /// Determine a reasonable step size that is guaranteed not to make a parameter value negative.
    ///
    /// An implementation of [1] equation 8.21
    ///
    /// [1] 4.3 Equation 8.21, ignoring 8.20 requirement
    /// same step is taken in primal and dual spaces
    /// alpha0 is basically beta3 from [1] Table 8.1, but instead of beta3
    /// the value 1 is used in Mehrota corrector and initial point correction
    fn get_step_size(&self, delta: &Delta<F>, alpha0: F) -> F {
        let min = |default: F, d_x: &F, x: &F| {
            if *d_x < F::zero() {
                default.min(*x / -*d_x)
            } else {
                default
            }
        };
        let alpha_x = Zip::from(&delta.d_x).and(&self.x).fold(F::one(), min);
        let alpha_z = Zip::from(&delta.d_z).and(&self.z).fold(F::one(), min);
        let alpha_tau = min(F::one(), &delta.d_tau, &self.tau);
        let alpha_kappa = min(F::one(), &delta.d_kappa, &self.kappa);

        F::one()
            .min(alpha_x)
            .min(alpha_tau)
            .min(alpha_z)
            .min(alpha_kappa)
            * alpha0
    }

    #[inline]
    /// [1] Equation 8.9
    fn do_step(self, delta: &Delta<F>, alpha: F, ip: bool) -> FeasiblePoint<F> {
        let x = &self.x + &(&delta.d_x * alpha);
        let y = &self.y + &(&delta.d_y * alpha);
        let z = &self.z + &(&delta.d_z * alpha);
        let tau = self.tau + delta.d_tau * alpha;
        let kappa = self.kappa + delta.d_kappa * alpha;

        // initial point
        // [1] 4.4
        // Formula after 8.23 takes a full step regardless if this will
        // take it negative
        if ip {
            FeasiblePoint {
                x: x.mapv(|e| e.max(F::one())),
                y,
                z: z.mapv(|e| e.max(F::one())),
                tau: tau.max(F::one()),
                kappa: kappa.max(F::one()),
                ..self
            }
        } else {
            FeasiblePoint {
                x,
                y,
                z,
                tau,
                kappa,
                ..self
            }
        }
    }

    /// Search directions for x,y,z, tau, kappa as defined in [1]
    /// Performs a predictor and corrector step, reusing a factorized system of equations
    fn get_delta(
        &self,
        problem: &Problem<F>,
        solver_type: &EquationSolverType,
        ip: bool,
    ) -> Result<Delta<F>, LinearProgramError<F>> {
        let n_x = self.x.len();

        // [1] Section 4.4
        let mut gamma = if ip { F::one() } else { F::zero() };
        let mut eta = if ip { F::one() } else { F::one() - gamma };
        // [1] Equation 8.8
        let r_P = &(&problem.b * self.tau) - &problem.A.dot(&self.x);
        let r_D = &(&problem.c * self.tau) - &problem.A.t().dot(&self.y) - &self.z;
        let r_G = problem.c.dot(&self.x) - problem.b.t().dot(&self.y) + self.kappa;
        let mu = (self.x.dot(&self.z) + self.tau * self.kappa) / F::cast(n_x + 1);

        //  Assemble M from [1] Equation 8.31
        let Dinv = &self.x / &self.z;
        let M = problem
            .A
            .dot(&(&Dinv.clone().insert_axis(ndarray::Axis(1)) * &problem.A.t()));

        let initial_solver = solver_type
            .build(&M)
            .or(Err(LinearProgramError::NumericalProblem))?;

        let rhat = Rhat::predictor(&r_P, &r_D, r_G, eta, self, gamma, mu);
        let (predictor_delta, predictor_solver) =
            Delta::compute(self, &rhat, problem, initial_solver, &Dinv, &M)?;

        // [1] 8.12 and "Let alpha be the maximal possible step..." before 8.23
        let alpha = self.get_step_size(&predictor_delta, F::one());
        gamma = update_gamma(ip, alpha);
        eta = if ip { F::one() } else { F::one() - gamma };
        let rhat = Rhat::corrector(
            &r_P,
            &r_D,
            r_G,
            eta,
            self,
            &predictor_delta,
            gamma,
            mu,
            alpha,
            ip,
        );
        let (delta, _) = Delta::compute(self, &rhat, problem, predictor_solver, &Dinv, &M)?;

        Ok(delta)
    }
}

struct Rhat<F> {
    p: Array1<F>,
    d: Array1<F>,
    g: F,
    xs: Array1<F>,
    tk: F,
}
impl<F: Float> Rhat<F> {
    fn predictor(
        r_P: &Array1<F>,
        r_D: &Array1<F>,
        r_G: F,
        eta: F,
        point: &FeasiblePoint<F>,
        gamma: F,
        mu: F,
    ) -> Rhat<F> {
        // Reference [1] Eq. 8.6
        // Reference [1] Eq. 8.7
        Rhat {
            p: r_P * eta,
            d: r_D * eta,
            g: r_G * eta,
            xs: &(&point.x * -F::one()) * &point.z + gamma * mu,
            tk: gamma * mu - point.tau * point.kappa,
        }
    }
    fn corrector(
        r_P: &Array1<F>,
        r_D: &Array1<F>,
        r_G: F,
        eta: F,
        point: &FeasiblePoint<F>,
        delta: &Delta<F>,
        gamma: F,
        mu: F,
        alpha: F,
        ip: bool,
    ) -> Rhat<F> {
        // Reference [1] Eq. 8.6

        let (rhatxs, rhattk) = if ip {
            // Reference [1] Eq. 8.23
            (
                &(&point.x * -F::one()) * &point.z - &(&delta.d_x * &delta.d_z) * alpha.powi(2)
                    + (F::one() - alpha) * gamma * mu,
                (F::one() - alpha) * gamma * mu
                    - point.tau * point.kappa
                    - alpha.powi(2) * delta.d_tau * delta.d_kappa,
            )
        } else {
            (
                // Reference [1] Eq. 8.13
                &(&point.x * -F::one()) * &point.z + gamma * mu - &(&delta.d_x * &delta.d_z),
                gamma * mu - point.tau * point.kappa - delta.d_tau * delta.d_kappa,
            )
        };
        Rhat {
            p: r_P * eta,
            d: r_D * eta,
            g: r_G * eta,
            xs: rhatxs,
            tk: rhattk,
        }
    }
}
struct Problem<F> {
    A: Array2<F>,
    b: Array1<F>,
    c: Array1<F>,
    c0: F,
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

#[inline]
fn update_gamma<F: Float>(ip: bool, alpha: F) -> F {
    if ip {
        // initial point - see [1] 4.4
        F::cast(10)
    } else {
        // predictor-corrector, [1] definition after 8.12
        let beta1 = F::cast(0.1); // [1] pg. 220 (Table 8.1)
        (F::one() - alpha).powi(2) * beta1.min(F::one() - alpha)
    }
}

enum EquationsSolver<F> {
    Cholesky(Array2<F>),
    Inv(QRDecomp<F, OwnedRepr<F>>),
    LstSq(QRDecomp<F, OwnedRepr<F>>, bool),
}

impl<F: Float> EquationsSolver<F> {
    fn cholesky(A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorizeh_into()?;
        #[cfg(not(feature = "blas"))]
        let factor = A.cholesky()?;
        Ok(EquationsSolver::Cholesky(factor))
    }
    fn inv(A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorize_into()?;
        #[cfg(not(feature = "blas"))]
        let factor = A.qr()?;
        Ok(EquationsSolver::Inv(factor)) // TODO can do with buffering, maybe QR
    }
    fn least_squares(A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
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
    fn solve(&mut self, b: ArrayView1<F>) -> Result<Array1<F>, LinalgError> {
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
}

/// Attempt to solve a system of equations with the specified solver. If solving fails, retry with the next solver.
/// If all solvers fail, a solver failed to be even created or we observe NaN in the output, fail with a NumericalProblem
/// error.
fn solve_round<F: Float>(
    Dinv: &Array1<F>,
    problem: &Problem<F>,
    mut solver: EquationsSolver<F>,
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
        sym_solve(Dinv, &problem.A, &problem.c, &problem.b, &mut solver),
        sym_solve(
            Dinv,
            &problem.A,
            &(&rhat.d - &(&rhat.xs / x)),
            &rhat.p,
            &mut solver,
        ),
    ) {
        if p.fold(false, |acc, e| acc || e.is_nan()) || q.fold(false, |acc, e| acc || e.is_nan()) {
            return Err(LinearProgramError::NumericalProblem);
        }
        return Ok((p, q, u, v, solver));
    }
    // Solving failed due to numerical problems.
    // Usually this doesn't happen. If it does, it happens when
    // there are redundant constraints or when approaching the
    // solution. If so, change solver.
    match solver {
        EquationsSolver::Cholesky(_) => solve_round(
            Dinv,
            problem,
            EquationsSolver::inv(M).or(Err(LinearProgramError::NumericalProblem))?,
            x,
            rhat,
            M,
        ),
        EquationsSolver::Inv(_) => solve_round(
            Dinv,
            problem,
            EquationsSolver::least_squares(M).or(Err(LinearProgramError::NumericalProblem))?,
            x,
            rhat,
            M,
        ),
        EquationsSolver::LstSq(..) => Err(LinearProgramError::NumericalProblem),
    }
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
pub enum EquationSolverType {
    Cholesky,
    Inverse,
    LeastSquares,
}
impl EquationSolverType {
    fn build<F: Float>(&self, A: &Array2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        match self {
            EquationSolverType::Cholesky => EquationsSolver::cholesky(A),
            EquationSolverType::Inverse => EquationsSolver::inv(A),
            EquationSolverType::LeastSquares => EquationsSolver::least_squares(A),
        }
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

#[derive(Error, Debug)]
pub enum LinearProgramError<F: Debug> {
    #[error("The problem is unconstrained, meaning the solution is the all-zeros vector if `c` is nonnegative, or unbounded otherwise.")]
    Unconstrained,
    #[error("The solver encountered numerical problems it could not recover from. Likely causes are linearly dependent constraints or variables whose scale differs by multiple orders of magnitude.")]
    NumericalProblem,
    #[error("A parameter was set to an invalid value. Alpha0 must be between 0 and 1 (exclusive) and the tolerance must be nonnegative.")]
    InvalidParameter,
    #[error("The dimensions of your cost- and constraint arrays do not align.")]
    IncompatibleInputDimensions,
    #[error("The solver finished successfully, it appears that the problem is infeasible.")]
    Infeasible,
    #[error("The solver finished successfully, it appears that your problem is unbounded.")]
    Unbounded,
    #[error("The solver failed to converge within the maximum number of iterations. Best solution after the final iteration:\n{0:#?}")]
    IterationLimitExceeded(Array1<F>),
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
