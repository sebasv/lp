#![allow(non_snake_case)]
use super::{delta::Delta, feasible_point::FeasiblePoint};

use crate::float::Float;

use ndarray::Array1;

pub(crate) struct Rhat<F> {
    pub(crate) p: Array1<F>,
    pub(crate) d: Array1<F>,
    pub(crate) g: F,
    pub(crate) xs: Array1<F>,
    pub(crate) tk: F,
}

impl<F: Float> Rhat<F> {
    pub(crate) fn predictor(
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn corrector(
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
