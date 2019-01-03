# FQS: Fast Quartic solver

Python function for computing roots of a quartic (4th order polynomial).


# Features

 * Computes both real and complex roots of one or many quartics
 * If roots are all distinct and real a fast closed-form analytical solution is used
 * If some of the roots are complex or there are multiple roots, a numerical eigensolver is used
 * Function does not use `for` loops, therefore it is much faster when a large set of quartic roots are needed.
 
 
 # Requirements
 
 Python 3.6+  
 Numpy 1.8.0.+
 
 
 # Usage
 
 See [`test_quartic_solver.py`](test_quartic_solver.py) for examples.
 
 Bascially, the function can be used as follows:
 
 ```
 import fqs
 roots, real_idx = fqs.quartic_roots(p)
 ```
 
 or
 
  ```
 from fqs import quartic_roots
 roots, real_idx = quartic_roots(p)
 ```

 where `p` is an array of polynomial coefficients of the form:
 ```
 p[0]*x^4 + p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4] = 0
 ```
 
 `roots` is an array of roots,  
 `real_idx` is a boolean array, where `True` denotes indices of all real roots.
 
 Stacked array of coefficients are allowed, which means that `p` may have size (5,), (5, _M_) or (_M_, 5), where _M_ > 0 is the number of polynomials. Consequently, `roots` will have size (_M_, 4).

 
 # FAQ


 > Why not simply use `numpy.roots` for all polynomials?
 
 If only one set of quartic roots are needed, then, by all means, use it. However, `numpy.roots` alows only rank-1 arrays, which means that if a large set of roots are needed, `numpy.roots` must be placed inside a `for` loop, which significantly slows down the computation (several orders of magnitude in comparison to `fqs`).
 
 > Why not use `numpy.linalg.eigvals` for all polynomials?
 
 The analytical closed-form solution is about 20-30 times faster than the numerical eigensolver when all four roots are distinct and real.
 
 # License

[MIT license](LICENSE)


