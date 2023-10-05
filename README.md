 A pure-Rust Interior Point solver for linear programs with equality and inequality constraints.

 The algorithm is heavily based on http://dx.doi.org/10.1007/978-1-4757-3216-0_8 (full text published free of charge on ResearchGate),
 and [the implementation thereof in SciPy](https://github.com/scipy/scipy/blob/4fc97887b5b6bc94af4b9f85fe8d10a2c62d6082/scipy/optimize/_linprog_ip.py).

 # Linear programs

 A linear program is a mathematical optimization problem defined as (using `@` as the dot product):

 ```text
    min_x c @ x
    st A_eq @ x == b_eq
       A_ub @ x <= b_ub
              x >= 0
 ```



 # Example
 ```rust
 use lp::{LinearProgram, SolverType};
 use approx::assert_abs_diff_eq;
 use ndarray::array;


 let A_ub = array![[-3f64, 1.], [1., 2.]];
 let b_ub = array![6., 4.];
 let A_eq = array![[1., 1.]];
 let b_eq = array![1.];
 let c = array![-1., 4.];

 let res = LinearProgram::new(c.view())
     // If you define neither equality nor inequality constraints, 
     // the problem returns as unconstrained.
     .ub(A_ub.view(), b_ub.view())
     .eq(A_eq.view(), b_eq.view())
     // The following are the default values you can overwrite.
     // You may omit any option for which the default is good enough for you
     .solver_type(SolverType::Cholesky)
     .tol(1e-8)
     .disp(false)
     .ip(true)
     .alpha0(0.99995)
     .max_iter(1000)
     .build()
     .unwrap()
     .try_solve()
     .unwrap();

 assert_abs_diff_eq!(res.x, array![1., 0.], epsilon = 1e-6);
 ```