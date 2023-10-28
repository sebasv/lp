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
pub trait Float: Lapack {
    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }
}

impl Float for f64 {}
impl Float for f32 {}
