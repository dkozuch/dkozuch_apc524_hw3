'''newton.py

Implementation of a Newton-Raphson root-finder.

'''

import numpy as np
import functions as F
import warnings

class Newton(object):
	"""Newton objects have a solve() method for finding roots of f(x)
	using Newton's method. x and f can both be vector-valued.

	"""
	
	def __init__(self, f, tol=1.e-6, maxiter=1000, dx=1.e-6, max_radius=np.inf, Df=False):
		"""Parameters:
		
		f: the function whose roots we seek. Can be scalar- or
		vector-valued. If the latter, must return an object of the
		same "shape" as its input x

		tol, maxiter: iterate until ||f(x)|| < tol or until you
		perform maxiter iterations, whichever happens first

		dx: step size for computing approximate Jacobian
		
		max_radius: maximum euclidean distance to search from x0
		
		Df: function representing the Jacobian of f; if set to zero the method will numerically approximate the Jacobian

		"""
		self._f = f
		self._tol = tol
		self._maxiter = maxiter
		self._dx = dx
		self._max_radius = max_radius
		self._Df = Df

	def solve(self, x0):
		"""Determine a solution of f(x) = 0, using Newton's method, starting
		from initial guess x0. x0 must be scalar or 1D-array-like. If
		x0 is scalar, return a scalar result; if x0 is 1D, return a 1D
		numpy array.

		"""
		x = x0
		for i in range(self._maxiter):
			fx = self._f(x)
			# linalg.norm works fine on scalar inputs
			if np.linalg.norm(fx) < self._tol:
				return x
			x = self.step(x, fx)
			
			if np.abs(np.linalg.norm(x-x0)) > self._max_radius:
				raise ValueError('The algorithm made a step that exceeded the specified maximum radius (maxr = '+str(self._max_radius)+'). Either specify a different maxr or different intial condition.') 
			
			if i == self._maxiter - 1:
				warnings.warn('Maximum iterations reached, but method has not converged to the requested tolerance',RuntimeWarning)
		return x

	def step(self, x, fx=None):
		"""Take a single step of a Newton method, starting from x. If the
		argument fx is provided, assumes fx = f(x).

		"""
		if fx is None:
			fx = self._f(x)

		#allow the user to provide the functional form of the Jacobian as Df; if Df is set to zero, we approximate numerically
		#because Df could be a scalar or a matrix, we have to test isscalar first
		if (np.isscalar(self._Df) and self._Df == 0) or (not np.isscalar(self._Df) and not np.any(self._Df)):
			Df_x = F.approximateJacobian(self._f, x, self._dx)
		else:
			Df_x = self._Df(x)
		
		#check for zero slope
		if np.sum(Df_x) == 0:
			raise ValueError('Algorithm encountered point with zero slope. Maybe try a different intial condition?')

		h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))
		
		if np.isscalar(x):
			h = np.asscalar(h)

		return x - h
