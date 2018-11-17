import numpy as np

def approximateJacobian(f, x, dx=1e-6):
    """Calculate a numerical approximation of the Jacobian Df(x).

    Parameters:
    
    f: a function that takes x as input. f should return the same sort
    of object that x is, i.e. if x is a scalar (not a numpy array), f
    should return a scalar; if x is a numpy array, f should return a
    numpy array *of the same shape*; if x is a numpy matrix, f should
    return a numpy matrix *of the same shape*.

    x: the input at which to approximate the Jacobian of f. Although
    it doesn't test for this explicitly, this routine assumes that x
    is one of: (i) a scalar, (ii) a 1D numpy array of shape (N,),
    (iii) a 2D numpy array of shape (N,1), or (iv) a numpy matrix of
    shape (N,1)

    Returns:
    
    Df_x: a numerical approximation to the Jacobian of f at x.  If x
    is a scalar, Df_x is a scalar. If x is something "array-like" of
    length N, then Df_x is an NxN numpy matrix.

    """
    #dx must be positive and greater than zero
    if dx <= 0:
        raise ValueError('dx must be positive non-zero')

    # Evaluate f(x) up front, since we'll need this value in multiple
    # places
    fx = f(x)

    # First, handle the case in which x is a scalar (i.e. not
    # array-like, just a plain number)

    #use symetric differentiation for higher numerical accuracy
    if np.isscalar(x):
        return (f(x + 0.5*dx) - f(x - 0.5*dx)) / float(dx)

    # For non-scalar case
    # Let's initialize Df_x to be an NxN numpy
    # matrix of zeros:
    N = x.size
    Df_x = np.matrix(np.zeros((N,N)))

    h = np.zeros_like(x)
    # We allocate this "vector of zeros" just once (to be
    # memory-efficient). Below, we're going to iterate over the
    # columns of Df_x and populate them with something nonzero.  As we
    # handle the ith column of Df_x, we'll flip the ith slot of h from
    # 0.0 to dx and back, so that during the ith iteration, h is
    # nonzero only in the ith slot.

    # ith column of the Jacobian consists of partials of f with
    # respect to x_i, so the difference quotient should involve
    # evaluating f at (x_1, x_2, ... x_i + delta_x_i, x_{i+1}, ...,
    # x_N). That's the same as evaluating f at (x + h), where h = (0,
    # 0,... dx [in ith slot], 0, ..., 0).  Addition on numpy
    # arrays/matrices happens elementwise.
    for i in range(x.size): # Could also have said range(x.size)
        h[i] = dx
        # Replace ith col of Df_x with difference quotient
        Df_x[:,i] = (f(x + 0.5*h) - (f(x - 0.5*h))) / float(dx)
        # Reset h[i] to 0
        h[i] = 0

    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 5x + 6,
    and evaluate p(3):

    p = Polynomial([6, 5, 1])

    p(3)

    """
    def __init__(self, coeffs):
        """In coeffs, index = degree of that coefficient"""
        self._coeffs = coeffs

    # The __repr__ method tells objects what to do when fed into the
    # print() function
    def __repr__(self):
        # Read up on the join() method of string objects. In this
        # case, we're calling the join() method of the string ','
        # consisting of a single comma.
        coeffstr = ",".join([str(x) for x in self._coeffs])
        # Read up on Python string formatting. I'm avoiding the newer
        # "format-strings" introduced in Python 3.7
        return "Polynomial([{}])".format(coeffstr)

    def _f(self,x):
        # We worked out this algorithm in lecture...
        ans = 0
        # Iterate from highest to lowest degree
        for c in reversed(self._coeffs):
            ans = x*ans + c
        return ans

    # Instances of classes that have a defined __call__ method are
    # themselves callable, as if they were functions
    def __call__(self, x):
        return self._f(x)

