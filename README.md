 A pure-Rust Interior Point solver for linear programs with equality and inequality constraints.

 The algorithm is heavily based on the [MOSEK](http://dx.doi.org/10.1007/978-1-4757-3216-0_8) solver ([open-access link](https://www.researchgate.net/publication/243774586_The_Mosek_Interior_Point_Optimizer_for_Linear_Programming_An_Implementation_of_the_Homogeneous_Algorithm))
, and [the implementation thereof in SciPy](https://github.com/scipy/scipy/blob/4fc97887b5b6bc94af4b9f85fe8d10a2c62d6082/scipy/optimize/_linprog_ip.py).

 # Linear programs

 A linear program is a mathematical optimization problem defined as (using `'` as the dot product):

 ```text
    min_x c ' x
    st A_eq ' x == b_eq
       A_ub ' x <= b_ub
              x >= 0
 ```



 # Example
 ```rust
 use lp::prelude::*;
 use approx::assert_abs_diff_eq;
 use ndarray::array;

 let A_ub = array![[-3f64, 1.], [1., 2.]];
 let b_ub = array![6., 4.];
 let A_eq = array![[1., 1.]];
 let b_eq = array![1.];
 let c = array![-1., 4.];

 let res = Problem::target(&c)
     .ub(&A_ub, &b_ub)
     .eq(&A_eq, &b_eq)
     .build()
     .unwrap();

 let solver = InteriorPoint::default();

 let solution = solver.solve(&problem);

 assert_abs_diff_eq!(solution.x, array![1., 0.], epsilon = 1e-6);
 ```