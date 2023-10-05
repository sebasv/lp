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
    Dinv: ArrayView1<F>,
    A: ArrayView2<F>,
    r1: ArrayView1<F>,
    r2: ArrayView1<F>,
    solver: &mut EquationsSolver<F>,
) -> Result<(Array1<F>, Array1<F>), LinalgError> {
    let r = &r2 + &A.dot(&(&Dinv * &r1));
    let v = solver.solve(r.view())?;
    // [1] 8.32
    let u = &Dinv * &(&A.t().dot(&v) - &r1);
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

    fn indicators(&self, problem: &Problem<F>) -> (F, F, F, F, F, F) {
        let obj = problem.c.dot(&(&self.x / self.tau)) + problem.c0;
        let rho_A = (problem.c.t().dot(&self.x) - problem.b.t().dot(&self.y)).abs()
            / (self.tau + problem.b.t().dot(&self.y).abs());
        let residuals = self.residuals(problem);
        (
            residuals.rho_p / self.initial_residuals.rho_p.max(F::one()),
            residuals.rho_d / self.initial_residuals.rho_d.max(F::one()),
            rho_A,
            residuals.rho_g / self.initial_residuals.rho_g.max(F::one()),
            residuals.rho_mu / self.initial_residuals.rho_mu,
            obj,
        )
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
    fn do_step(&self, delta: &Delta<F>, alpha0: F, ip: bool) -> FeasiblePoint<F> {
        let alpha = if ip {
            F::one()
        } else {
            // [1] Section 4.3
            self.get_step_size(delta, alpha0)
        };
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
                y: y,
                z: z.mapv(|e| e.max(F::one())),
                tau: tau.max(F::one()),
                kappa: kappa.max(F::one()),
                initial_residuals: self.initial_residuals,
            }
        } else {
            FeasiblePoint {
                x,
                y,
                z,
                tau,
                kappa,
                initial_residuals: self.initial_residuals,
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
        let r_P = &(&b * tau) - &A.dot(&x);
        let r_D = &(&c * tau) - &A.t().dot(&y) - z;
        let r_G = c.dot(&x) - b.t().dot(&y) + kappa;
        let mu = (x.dot(&z) + tau * kappa) / F::cast(n_x + 1);

        //  Assemble M from [1] Equation 8.31
        let Dinv = &x / &z;
        let M = A.dot(&(&Dinv.clone().insert_axis(ndarray::Axis(1)) * &A.t()));

        let mut solver = solver_type
            .build(M.view())
            .or(Err(LinearProgramError::NumericalProblem))?;

        let (mut alpha, mut d_tau, mut d_kappa) = (F::zero(), F::zero(), F::zero());
        let mut d_x: Array1<F> = Array1::zeros(x.dim());
        let mut d_y: Array1<F> = Array1::zeros(x.dim());
        let mut d_z: Array1<F> = Array1::zeros(x.dim());

        for step in [Step::Predictor, Step::Corrector] {
            // Reference [1] Eq. 8.6
            let rhatp = &r_P * eta;
            let rhatd = &r_D * eta;
            let rhatg = r_G * eta;

            let (rhatxs, rhattk) = match step {
                Step::Predictor => {
                    // Reference [1] Eq. 8.7
                    (
                        &(&x * -F::one()) * &z + gamma * mu,
                        gamma * mu - tau * kappa,
                    )
                }
                Step::Corrector => {
                    if ip {
                        // Reference [1] Eq. 8.23
                        (
                            &(&x * -F::one()) * &z - &(&d_x * &d_z) * alpha.powi(2)
                                + (F::one() - alpha) * gamma * mu,
                            (F::one() - alpha) * gamma * mu
                                - tau * kappa
                                - alpha.powi(2) * d_tau * d_kappa,
                        )
                    } else {
                        (
                            // Reference [1] Eq. 8.13
                            &(&x * -F::one()) * &z + gamma * mu - &(&d_x * &d_z),
                            gamma * mu - tau * kappa - d_tau * d_kappa,
                        )
                    }
                }
            };

            let (p, q, u, v);
            (p, q, u, v, solver) = solve_round(
                Dinv.view(),
                problem.A,
                problem.b,
                problem.c,
                solver,
                self.x,
                rhatd.view(),
                rhatxs.view(),
                rhatp.view(),
                M.view(),
            )?;
            // [1] Results after 8.29
            d_tau = (rhatg + F::one() / tau * rhattk - (-c.dot(&u) + b.dot(&v)))
                / (F::one() / tau * kappa + (-c.dot(&p) + b.dot(&q)));
            d_x = &u + &(&p * d_tau);
            d_y = v + q * d_tau;

            // [1] Relations between  after 8.25 and 8.26
            d_z = (rhatxs - &z * &d_x) / x;
            d_kappa = F::one() / tau * (rhattk - kappa * d_tau);

            // [1] 8.12 and "Let alpha be the maximal possible step..." before 8.23
            alpha = get_step_size(
                x,
                d_x.view(),
                z,
                d_z.view(),
                tau,
                d_tau,
                kappa,
                d_kappa,
                F::one(),
            );
            gamma = update_gamma(ip, alpha);
            eta = if ip { F::one() } else { F::one() - gamma };
        }
        Ok((d_x, d_y, d_z, d_tau, d_kappa))
    }
}
struct Problem<F> {
    A: Array2<F>,
    b: Array1<F>,
    c: Array1<F>,
    c0: F,
}
enum Step {
    Predictor,
    Corrector,
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
    fn cholesky(A: ArrayView2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorizeh_into()?;
        #[cfg(not(feature = "blas"))]
        let factor = A.cholesky()?;
        Ok(EquationsSolver::Cholesky(factor))
    }
    fn inv(A: ArrayView2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorize_into()?;
        #[cfg(not(feature = "blas"))]
        let factor = A.qr()?;
        Ok(EquationsSolver::Inv(factor)) // TODO can do with buffering, maybe QR
    }
    fn least_squares(A: ArrayView2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        #[cfg(feature = "blas")]
        let factor = A.factorize_into()?;
        #[cfg(not(feature = "blas"))]
        let (factor, transposed) = if A.nrows() >= A.ncols() {
            (A.qr()?, false)
        } else {
            (A.reversed_axes().qr()?, true)
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
    Dinv: ArrayView1<F>,
    A: ArrayView2<F>,
    b: ArrayView1<F>,
    c: ArrayView1<F>,
    mut solver: EquationsSolver<F>,
    x: ArrayView1<F>,
    rhatd: ArrayView1<F>,
    rhatxs: ArrayView1<F>,
    rhatp: ArrayView1<F>,
    M: ArrayView2<F>,
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
        sym_solve(Dinv, A, c, b, &mut solver),
        sym_solve(
            Dinv,
            A,
            (&rhatd - &(&rhatxs / &x)).view(),
            rhatp,
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
            A,
            b,
            c,
            EquationsSolver::inv(M).or(Err(LinearProgramError::NumericalProblem))?,
            x,
            rhatd,
            rhatxs,
            rhatp,
            M,
        ),
        EquationsSolver::Inv(_) => solve_round(
            Dinv,
            A,
            b,
            c,
            EquationsSolver::least_squares(M).or(Err(LinearProgramError::NumericalProblem))?,
            x,
            rhatd,
            rhatxs,
            rhatp,
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
    fn build<F: Float>(&self, A: ArrayView2<F>) -> Result<EquationsSolver<F>, LinalgError> {
        match self {
            EquationSolverType::Cholesky => EquationsSolver::cholesky(A),
            EquationSolverType::Inverse => EquationsSolver::inv(A),
            EquationSolverType::LeastSquares => EquationsSolver::least_squares(A),
        }
    }
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
pub fn ip_hsd<F: Float>(
    A: ArrayView2<F>,
    b: ArrayView1<F>,
    c: ArrayView1<F>,
    c0: F,
    alpha0: F,
    maxiter: usize,
    disp: bool,
    tol: F,
    solver_type: &EquationSolverType,
    mut ip: bool,
) -> Result<(Array1<F>, usize), LinearProgramError<F>> {
    // default initial point
    let (m, n) = A.dim();
    let (mut x, mut y, mut z, mut tau, mut kappa) = get_blind_start(m, n);

    // [1] 4.5
    let (mut rho_p, mut rho_d, mut rho_A, mut rho_g, mut rho_mu, mut obj) =
        indicators(A, b, c, c0, x.view(), y.view(), z.view(), tau, kappa);

    if disp {
        display_iter(rho_p, rho_d, rho_g, F::zero(), rho_mu, obj, true);
    }

    for iteration in 0..maxiter {
        // Solve [1] 8.6 and 8.7/8.13/8.23
        let (d_x, d_y, d_z, d_tau, d_kappa) = get_delta(
            A,
            b,
            c,
            x.view(),
            y.view(),
            z.view(),
            tau,
            kappa,
            solver_type,
            ip,
        )?;
        let alpha;
        (x, y, z, tau, kappa, alpha) = do_step(
            x.view(),
            y.view(),
            z.view(),
            tau,
            kappa,
            d_x.view(),
            d_y.view(),
            d_z.view(),
            d_tau,
            d_kappa,
            alpha0,
            ip,
        );
        ip = false;

        // [1] 4.5
        (rho_p, rho_d, rho_A, rho_g, rho_mu, obj) =
            indicators(A, b, c, c0, x.view(), y.view(), z.view(), tau, kappa);

        if disp {
            display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, false);
        }

        // [1] 4.5
        let inf1 = rho_p < tol && rho_d < tol && rho_g < tol && tau < tol * kappa.max(F::one());
        let inf2 = rho_mu < tol && tau < tol * kappa.min(F::one());
        if inf1 || inf2 {
            // [1] Lemma 8.4 / Theorem 8.3
            if b.t().dot(&y) > tol {
                return Err(LinearProgramError::Infeasible);
            } else {
                return Err(LinearProgramError::Unbounded);
            }
        } else if rho_p < tol && rho_d < tol && rho_A < tol {
            // [1] Statement after Theorem 8.2
            let x_hat = x / tau;
            return Ok((x_hat, iteration));
        }
    }
    Err(LinearProgramError::IterationLimitExceeded(x / tau))
}

/// print progress of convergence criteria to stdout
fn display_iter<F: Float>(
    rho_p: F,
    rho_d: F,
    rho_g: F,
    alpha: F,
    rho_mu: F,
    obj: F,
    display_header: bool,
) {
    if display_header {
        println!(" rho_p     \t rho_d     \t rho_g     \t alpha     \t rho_mu    \t obj       ");
    }
    println!("{rho_p:3.8}\t{rho_d:3.8}\t{rho_g:3.8}\t{alpha:3.8}\t{rho_mu:3.8}\t{obj:8.3}");
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
        let (A, b, c) = lp_problem_to_slack_form(self.c, self.ub, self.eq)?;
        if self.alpha0 <= F::zero() || self.alpha0 >= F::one() {
            return Err(LinearProgramError::InvalidParameter);
        }
        if self.tol <= F::zero() {
            return Err(LinearProgramError::InvalidParameter);
        }
        let n_slack = c.len() - self.c.len();
        Ok(LinearProgram {
            A,
            b,
            c,
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
    A: Array2<F>,
    b: Array1<F>,
    c: Array1<F>,
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
    /// let res = LinearProgram::new(c.view())
    ///     .ub(A_ub.view(), b_ub.view())
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn new(c: ArrayView1<F>) -> LinearProgramInput<F> {
        LinearProgramInput::new(c)
    }

    /// Attempt to solve the linear program.
    pub fn solve(&self) -> Result<OptimizeResult<F>, LinearProgramError<F>> {
        let c0 = F::zero(); // constant term in the objective function if we add presolving

        let (x_slack, iteration) = ip_hsd(
            self.A.view(),
            self.b.view(),
            self.c.view(),
            c0,
            self.alpha0,
            self.max_iter,
            self.disp,
            self.tol,
            &self.solver_type,
            self.ip,
        )?;
        // Eliminate artificial variables, re-introduce presolved variables, etc.
        let x = x_slack.slice(s![..x_slack.len() - self.n_slack]).to_owned();
        let fun = self.c.dot(&x_slack);
        Ok(OptimizeResult { x, fun, iteration })
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
fn lp_problem_to_slack_form<F: Float>(
    c: ArrayView1<F>,
    ub: Option<(ArrayView2<F>, ArrayView1<F>)>,
    eq: Option<(ArrayView2<F>, ArrayView1<F>)>,
) -> Result<(Array2<F>, Array1<F>, Array1<F>), LinearProgramError<F>> {
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
    Ok((A, b, c))
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

        let res = LinearProgram::new(c.view())
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

        let res = LinearProgram::new(c.view())
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

        let res = LinearProgram::new(c.view())
            .ub(A_ub.view(), b_ub.view())
            .build()
            .unwrap()
            .solve()
            .unwrap();

        assert_abs_diff_eq!(res.x, array![0.5, 0.0, 1.25], epsilon = 1e-6);
    }
}
