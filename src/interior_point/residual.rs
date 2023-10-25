use ndarray::Array1;

use crate::float::Float;
use crate::interior_point::linprog::Problem;
pub(crate) struct Residuals<F> {
    pub(crate) rho_p: F,
    pub(crate) rho_d: F,
    pub(crate) rho_g: F,
    pub(crate) rho_mu: F,
}

impl<F: Float> Residuals<F> {
    pub(crate) fn calculate(
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
