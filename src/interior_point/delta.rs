#![allow(non_snake_case)]
use crate::error::LinearProgramError;

use ndarray::Array2;

use super::equations_solver::EquationsSolver;

use super::feasible_point::FeasiblePoint;
use super::linprog::Problem;
use super::rhat::Rhat;

use crate::float::Float;

use ndarray::Array1;

pub(crate) struct Delta<F> {
    pub(crate) d_x: Array1<F>,
    pub(crate) d_y: Array1<F>,
    pub(crate) d_z: Array1<F>,
    pub(crate) d_tau: F,
    pub(crate) d_kappa: F,
}

impl<F: Float> Delta<F> {
    pub(crate) fn compute(
        point: &FeasiblePoint<F>,
        rhat: &Rhat<F>,
        problem: &Problem<F>,
        solver: EquationsSolver<F>,
    ) -> Result<(Delta<F>, EquationsSolver<F>), LinearProgramError<F>> {
        let (newton, new_solver) = solver.solve_newton_equations(problem, &point.x, rhat)?;
        // [1] Results after 8.29
        let d_tau = (rhat.g + F::one() / point.tau * rhat.tk
            - (-problem.c.dot(&newton.u) + problem.b.dot(&newton.v)))
            / (F::one() / point.tau * point.kappa
                + (-problem.c.dot(&newton.p) + problem.b.dot(&newton.q)));
        let d_x = newton.u + newton.p * d_tau;
        let d_y = newton.v + newton.q * d_tau;

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
