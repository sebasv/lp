#![allow(non_snake_case)]
use super::rhat::Rhat;

use ndarray::Array1;

use crate::error::LinearProgramError;

use super::newton_equations::EquationSolverType;

use ndarray::Zip;

use super::delta::Delta;

use crate::linear_program::Problem;

use super::residual::Residuals;

use crate::float::Float;

pub(crate) struct FeasiblePoint<F> {
    pub(crate) x: Array1<F>,
    pub(crate) y: Array1<F>,
    pub(crate) z: Array1<F>,
    pub(crate) tau: F,
    pub(crate) kappa: F,
    pub(crate) initial_residuals: Residuals<F>,
}

impl<F: Float> FeasiblePoint<F> {
    pub(crate) fn blind_start(problem: &Problem<F>) -> FeasiblePoint<F> {
        let (m, n) = problem.A().dim();
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
    pub(crate) fn residuals(&self, problem: &Problem<F>) -> Residuals<F> {
        Residuals::calculate(problem, &self.x, &self.y, &self.z, self.tau, self.kappa)
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
    pub(crate) fn get_step_size(&self, delta: &Delta<F>, alpha0: F) -> F {
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
    pub(crate) fn do_step(self, delta: &Delta<F>, alpha: F, ip: bool) -> FeasiblePoint<F> {
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
    pub(crate) fn get_delta(
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
        let r_P = &(problem.b() * self.tau) - &problem.A().dot(&self.x);
        let r_D = &(problem.c() * self.tau) - &problem.A().t().dot(&self.y) - &self.z;
        let r_G = problem.c().dot(&self.x) - problem.b().t().dot(&self.y) + self.kappa;
        let mu = (self.x.dot(&self.z) + self.tau * self.kappa) / F::cast(n_x + 1);

        let initial_solver = solver_type
            .build(self, problem)
            .or(Err(LinearProgramError::NumericalProblem))?;

        let rhat = Rhat::predictor(&r_P, &r_D, r_G, eta, self, gamma, mu);
        let (predictor_delta, predictor_solver) =
            Delta::compute(self, &rhat, problem, initial_solver)?;

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
        let (delta, _) = Delta::compute(self, &rhat, problem, predictor_solver)?;

        Ok(delta)
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
