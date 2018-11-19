#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import newton

class TestNewton(unittest.TestCase):
	
	def test_guessIsSolution(self):
		#if the intial guess, x0, is the root, don't continue
		#probably unecessary but doesn't hurt
		f = lambda x : 3.0*x + 6.0
		solver = newton.Newton(f, tol=1.e-15, maxiter=1)
		x = solver.solve(-2.0) #-2.0 is the root
		self.assertEqual(x, -2.0)

	def test_1D(self):
		#test 1D function
		f = lambda x : 3.0*x + 6.0
		solver = newton.Newton(f, tol=1.e-15, maxiter=10)
		x = solver.solve(2.0)
		self.assertEqual(x, -2.0)

		#test another 1D function, but has 2 roots
		f = lambda x : (x**2) - 100
		solver = newton.Newton(f, tol=1.e-15, maxiter=10)
		x = solver.solve(2.0)
		self.assertTrue(x == 10.0 or x == -10.0)
		
		x = solver.solve(-9.0)
		self.assertTrue(x == 10.0 or x == -10.0)

		#test function with no roots
		f = lambda x : (x**2) + 10
		solver = newton.Newton(f, tol=1.e-15, maxiter=10)
		with self.assertWarns(RuntimeWarning):
			solver.solve(2.0)

		#test function with zero derivative
		def f(x):
			return 1
		solver = newton.Newton(f, tol=1.e-15, maxiter=10)
		self.assertRaises(ValueError,solver.solve,2.0)


	def test_2D(self):
		
		A = np.matrix("1.0 2.0; 1.0 3.0")
		B = np.matrix("-1.0; 3.0")
		def f(x):
			return A*x + B

		solver = newton.Newton(f, tol=1.e-15, maxiter=10)
		x = solver.solve(np.matrix("2.0; 2.0"))
		npt.assert_array_almost_equal(x,np.matrix("9.0; -4.0"))
   
	def test_maxiter(self):
		#make sure runtime error is triggered if maxiter reached	 
		A = np.matrix("1.0 2.0; 1.0 3.0")
		B = np.matrix("-1.0; 3.0")
		def f(x):
			return A*x + B

		solver = newton.Newton(f, tol=1.e-15, maxiter=3) #choose small number of maxiter
		with self.assertWarns(RuntimeWarning):
			solver.solve(np.matrix("200.0; 200.0"))

if __name__ == "__main__":
	unittest.main()

