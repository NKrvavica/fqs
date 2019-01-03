# FQS: Fast Quartic solver

Python function for computing roots of a quartic (4th order polynomial).


# Features

 * Computes both real and complex roots of one or many quartics
 * If roots are all distinct and real a fast closed-form analytical solution is used
 * If come of the roots are complex or there are multiple roots, a numerical eigensolver is used
 * No for loops are used, therefore the function is very fast when roots of a large set of polynomials are needed.
 
 
 # Requirements
 
 Python 3.6+  
 Numpy 1.8.0.+
 
 
 # Usage
 
 See [`test_Quartic_solver.py`](test_quartic_solver.py) for examples.
 
 Function can be used as follows:
 
 ```
 import fqs
 roots, real_idx = fqs.quartic_roots(p)
 ```
 
 or
 
  ```
 from fqs import quartic_roots
 roots, real_idx = quartic_roots(p)
 ```

 where `p` is array of polynomial coefficients of the form:
 ```
 p[0]*x^4 + p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4] = 0
 ```
 
 `roots` is array of roots,  
 `real_idx` is boolean array, where `True` denotes indices of all real roots.
 
 Stacked array of coefficient are allowed, which means that `p` may have size (5,), (5, _M_) or (_M_, 5), where _M_ > 0 is the number of polynomials. Consequently, `roots` will have size (_M_, 4).

 
 # FAQ


 > Why not simply use `numpy.roots` for all polynomails?
 
 If only one set of quartic roots are need, then by all means use it. However, `numpy.roots` alows only rank-1 arrays, which mean that if a large set of roots are needed, `numpy.roots` must be placed inside a `for` loop, which significantly slows down the computation (several orders of magnitude in comparison to `fqs`).
 
 > Why not use `numpy.linalg.eigvals` for all polynomials?
 
 Analytical closed-form solution is about 20-30 times faster than the numerical eigensolver when all four roots are distinct and real.
 
 # License

[MIT license](LICENSE)


