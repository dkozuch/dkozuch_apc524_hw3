# Example script

#make sure newton.py is in the working directory and import the module
import newton

#define a simple function f
def f(x):
        return 5*x - 10

#instantiate the solver
solver = newton.Newton(f)

#solve for the root
root = solver.solve(25.0)

#print the answer
print(root)

