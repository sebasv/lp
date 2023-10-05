use ndarray::NdFloat;
use num_traits::NumCast;

pub trait Float: NdFloat {
    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }
}

impl Float for f64 {}
impl Float for f32 {}
