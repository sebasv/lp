use ndarray::NdFloat;
#[cfg(feature = "blas")]
use ndarray_linalg::Lapack;
use num_traits::NumCast;
#[cfg(not(feature = "blas"))]
pub trait Lapack {}

pub trait Float: NdFloat + Lapack {
    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }

    fn powi(self, n: i32) -> Self {
        num_traits::Float::powi(self, n)
    }
    fn abs(self) -> Self {
        num_traits::Float::abs(self)
    }
    fn sqrt(self) -> Self {
        num_traits::Float::sqrt(self)
    }
}

impl Float for f64 {}
impl Float for f32 {}
