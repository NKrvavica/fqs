# FQS: Fast Quartic and Cubic solver

A fast python function for computing roots of a quartic equation (4th order polynomial) and a cubic equation (3rd order polynomial).


# Features

 * The function is optimized for computing a large set of roots of 3rd and 4th order polynomials (cubic and quartic equations).
 * A closed-form analytical solutions of Ferrari and Cardano are used, which are several times faster than the integral `numpy.roots` or `numpy.linalg.eigvals` functions.
 * The algorithm is based on `numpy` to avoid using `for` loops when multiple polynomials are evaluated.
 
 
 # Requirements
 
 Python 3.6+  
 Numpy 1.8.0.+
 
 
 # Usage
 
See [test_cubic_roots.py](test_cubic_roots.py) and [test_quartic_roots.py](test_quartic_roots.py) for example on the usage and performance in comparison to `numpy.roots` and `numpy.linalg.eigvals`.

For quartic roots, the function can be used as follows:
 
 ```
 import fqs
 roots = fqs.quartic_roots(p)
 
 ```
 
 or
 
  ```
 from fqs import quartic_roots
 roots = quartic_roots(p)
 ```

 where `p` is an array of polynomial coefficients of the form:
 ```
 p[0]*x^4 + p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4] = 0
 ```
 
 and `roots` is an array of resulting four roots.  
 
 Stacked array of coefficients are allowed, which means that `p` may have size (5,) or (_M_, 5), where _M_ > 0 is the number of polynomials. Consequently, `roots` will have size (_M_, 4).

 
 For cubic roots, the function can be used as follows:
 
 ```
 import fqs
 roots = fqs.cubic_roots(p)
 
 ```
 
 or
 
  ```
 from fqs import cubic_roots
 roots = cubic_roots(p)
 ```

 where `p` is an array of polynomial coefficients of the form:
 ```
p[0]*x^3 + p[1]*x^2 + p[2]*x + p[3] = 0
 ```
 
 `roots` is an array of resulting three roots.  
 
 Stacked array of coefficients are allowed, which means that `p` may have size (4,) or (_M_, 4), where _M_ > 0 is the number of polynomials. Consequently, `roots` will have size (_M_, 3).


 
 # FAQ

 > Why not simply use `numpy.roots` for all polynomials?
 
 If roots of only one polynomial (cubic or quartic) are needed, then, by all means, use it. However, `numpy.roots` alows only rank-1 arrays, which means that if roots of a large set of polynomials are needed, `numpy.roots` must be placed inside a `for` loop, which significantly slows down the computation (for example, `fqs` solvers are ~130-150 times faster than `numpy.roots` inside a `for` loop, when 10 000 polynomials are evaluated).
 
 > Why not use `numpy.linalg.eigvals` for all polynomials?
 
 True, `numpy.linalg.eigvals` can evaluate roots of multiple polynomials via companion matrix; however, `fqs` solvers are based on analytical solutions, which are about 5-8 times faster than the numerical eigensolver.
 
 
 # License
 
[MIT license](LICENSE)


