use ndarray::NdFloat;
#[cfg(feature = "blas")]
use ndarray_linalg::Lapack;
use num_traits::NumCast;

#[cfg(not(feature = "blas"))]
pub trait Float: NdFloat {
    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }
}
#[cfg(feature = "blas")]
pub trait Float: Lapack + NdFloat {
    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }
    fn abs(self) -> Self {
        num_traits::Float::abs(self)
    }
    fn sqrt(self) -> Self {
        num_traits::Float::sqrt(self)
    }
    fn powi(self, n: i32) -> Self {
        num_traits::Float::powi(self, n)
    }
}

impl Float for f64 {}
impl Float for f32 {}
