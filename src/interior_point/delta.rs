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
        Dinv: &Array1<F>,
        M: &Array2<F>,
    ) -> Result<(Delta<F>, EquationsSolver<F>), LinearProgramError<F>> {
        let (p, q, u, v, new_solver) = solver.solve_round(Dinv, problem, &point.x, rhat, M)?;
        // [1] Results after 8.29
        let d_tau = (rhat.g + F::one() / point.tau * rhat.tk
            - (-problem.c.dot(&u) + problem.b.dot(&v)))
            / (F::one() / point.tau * point.kappa + (-problem.c.dot(&p) + problem.b.dot(&q)));
        let d_x = u + p * d_tau;
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
