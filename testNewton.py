#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import newton

class TestNewton(unittest.TestCase):
    
    def testLinear(self):
        # Just so you see it at least once, this is the lambda keyword
        # in Python, which allows you to create anonymous functions
        # "on the fly". As I commented in testFunctions.py, you can
        # define regular functions inside other
        # functions/methods. lambda expressions are just syntactic
        # sugar for that.  In other words, the line below is
        # *completely equivalent* under the hood to:
        #
        # def f(x):
        #     return 3.0*x + 6.0
        #
        # No difference.
        f = lambda x : 3.0*x + 6.0

        # Setting maxiter to 2 b/c we're guessing the actual root
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        # Equality should be exact if we supply *the* root, ergo
        # assertEqual rather than assertAlmostEqual
        self.assertEqual(x, -2.0)

    def test_guessIsSolution(self):
        #if the intial guess, x0, is the root, don't continue 
        f = lambda x : 3.0*x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(-2.0) #-2.0 is the root
        self.assertEqual(x, -2.0)

    def test_2D(self):
        
        A = np.matrix("1.0 2.0; 1.0 3.0")
        B = np.matrix("-1.0; 3.0")
        def f(x):
            p1 = A*x
            p2 = p1 + B
            return p2

        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(np.matrix("2.0; 2.0"))
        npt.assert_array_almost_equal(x,np.matrix("9.0; -4.0"))
        

if __name__ == "__main__":
    unittest.main()

    
