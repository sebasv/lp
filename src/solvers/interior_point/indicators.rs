#![allow(non_snake_case)]
use std::fmt::Display;

use crate::{float::Float, linear_program::Problem};

use super::feasible_point::FeasiblePoint;

pub(crate) struct Indicators<F> {
    /// primal infeasibility
    rho_p: F,
    /// dual infeasibility
    rho_d: F,
    /// number of significant digits in objective value
    rho_A: F,
    /// gap infeasibility
    rho_g: F,
    /// ???
    rho_mu: F,
    /// Primal objective value
    obj: F,
    /// Dual objective value
    bty: F,
}

impl<F: Display> Display for Indicators<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:3.8}\t{:3.8}\t{:3.8}\t{:3.8}\t{:8.3}",
            self.rho_p, self.rho_d, self.rho_g, self.rho_mu, self.obj
        )
    }
}

impl<F: Float> Indicators<F> {
    /// [1] 4.5
    pub(crate) fn from_point_and_problem(
        point: &FeasiblePoint<F>,
        problem: &Problem<F>,
    ) -> Indicators<F> {
        let obj = problem.c().dot(&(&point.x / point.tau)) + problem.c0();
        let bty = problem.b().t().dot(&point.y);
        let rho_A = Float::abs(problem.c().t().dot(&point.x) - bty)
            / (point.tau + Float::abs(problem.b().t().dot(&point.y)));
        let residuals = point.residuals(problem);
        Indicators {
            rho_p: residuals.rho_p / point.initial_residuals.rho_p.max(F::one()),
            rho_d: residuals.rho_d / point.initial_residuals.rho_d.max(F::one()),
            rho_A,
            rho_g: residuals.rho_g / point.initial_residuals.rho_g.max(F::one()),
            rho_mu: residuals.rho_mu / point.initial_residuals.rho_mu,
            obj,
            bty,
        }
    }
    /// Indicators show that infeasibility gaps are almost closed -> it does not get any better.
    fn smaller_than(&self, tol: F) -> bool {
        self.rho_p < tol && self.rho_d < tol && self.rho_A < tol
    }

    /// Indicators show that it does not get any better
    fn max_reduction_infeasibility_le(&self, tol: F) -> bool {
        self.rho_p < tol && self.rho_d < tol && self.rho_g < tol
    }

    pub(crate) fn status(&self, tau: F, kappa: F, tol: F) -> Status {
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
pub(crate) enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    Unfinished,
}
