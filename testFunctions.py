#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import functions as F

class TestFunctions(unittest.TestCase):
	def test_ApproxJacobian1(self):
		slope = 3.0
		# Yes, you can define a function inside a function/method. And
		# it has scope only within the method within which it's
		# defined (unless you return it to the outside world, which
		# you can do in Python with no need for anything like C's
		# malloc() or C++'s new() )
		def f(x):
			return slope * x + 5.0

		x0 = 2.0
		dx = 1.e-3
		Df_x = F.approximateJacobian(f, x0, dx)
		# self.assertEqual(Df_x.shape, (1,1)
		# If x and f are scalar-valued, Df_x should be, too
		self.assertTrue(np.isscalar(Df_x))
		self.assertAlmostEqual(Df_x, slope)

	def test_ApproxJacobian2(self):

		A = np.matrix("1.0 2.0; 3.0 4.0")

		def f(x):
			return A * x


		x0 = np.matrix("5.0; 6.0")
		dx = 1.e-6
		Df_x = F.approximateJacobian(f, x0, dx)

		# Make sure approximateJA
		self.assertEqual(Df_x.shape, (2,2))
		npt.assert_array_almost_equal(Df_x, A)

	   
		#test with a 3x3 array; shouldn't really be any different
		A = np.matrix("3.0 1.0 2.0; 9.0 0.0 12.0; 16.5 23.0 1.2")
		def f(x):
			return A*x

		x0 = np.matrix("5.0; 6.0; 17.2")
		dx = 1.e-6
		Df_x = F.approximateJacobian(f, x0, dx)
		npt.assert_array_almost_equal(Df_x, A)

	def test_dxZero(self):
		#make sure approximateJacobian raises error if dx is <= 0
		f = F.Polynomial([1,2,3,4])
		x0 = 2
		dx = 0
		self.assertRaises(ValueError,F.approximateJacobian,f=f,x=x0,dx=dx)

	def test_higherOrder(self):
		#test higher order expressions
	   f = F.Polynomial([1,2,3,4])
	   x0 = 2
	   dx = 1.e-6
	   Df_x = F.approximateJacobian(f, x0, dx)
	   self.assertAlmostEqual(Df_x,2+(3*2*x0)+(4*3*(x0**2)))

	def test_Polynomial(self):
		# p(x) = x^2 + 5x + 4
		p = F.Polynomial([4, 5, 1])
		# linspace(a, b, N) produces N equally spaced values from a to
		# b, including both a & b.
		for x in np.linspace(-2,2,11):
			self.assertAlmostEqual(p(x), 4 + 5*x + x**2)
		
if __name__ == '__main__':
	unittest.main()



