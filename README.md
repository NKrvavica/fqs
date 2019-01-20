# FQS: Fast Quartic and Cubic solver

A fast python function for computing roots of a quartic equation (4th order polynomial) and a cubic equation (3rd order polynomial).


# Features

 * The function is optimized for computing single or multiple roots of 3rd and 4th order polynomials (cubic and quartic equations).
 * A closed-form analytical solutions of Ferrari and Cardano are used for roots of cubic and quartic equations.
 * For a single or small number (<100) of polynomials, the algorithm is based on pure python with `numba` just-in-time compiler for faster cpu times.
 * For a large number of polynomials (>100), the algorithm is based on `numpy` to avoid using slower list comprehensions.
 * The solver is two order of magnitude faster than the integral `numpy.roots` inside a list comprehension, and several times faster than the `numpy.linalg.eigvals` function.
 
 A detailed documentation with cpu time analyses is available [here](On_computing_roots.md).
 
 
 # Requirements
 
 Python 3+, math, cmath, numpy, numba
 
 
 # Usage

All functions are found in `fqs.py`, which can be cloned to local folder.
 
See [test_cubic_roots.py](test_cubic_roots.py) and [test_quartic_roots.py](test_quartic_roots.py) for example on the usage and performance in comparison to `numpy.roots` and `numpy.linalg.eigvals`.

For quartic roots, the function can be used as follows:
 
 ```
 import fqs
 roots = fqs.quartic_roots(p)
 
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
 
 where `p` is an array of polynomial coefficients of the form:
 ```
p[0]*x^3 + p[1]*x^2 + p[2]*x + p[3] = 0
 ```
 
 `roots` is an array of resulting three roots.  
 
 Stacked array of coefficients are allowed, which means that `p` may have size (4,) or (_M_, 4), where _M_ > 0 is the number of polynomials. Consequently, `roots` will have size (_M_, 3).


 
 # FAQ

 > Why not simply use `numpy.roots` or `numpy.linalg.eigvals` for all polynomials?
 
For single polynomial, both quartic and cubic solvers are one order of magnitude faster than `numpy.roots` and `numpy.linalg.eigvals`. 
For large number of polynomials (>1_0000), both quartic and cubic solver are one order of magnitude faster than `numpy.linalg.eigvals` and two order of magnitude faster than `numpy.roots` inside a list comprehension.
 
 
 # License
 
[MIT license](LICENSE)


