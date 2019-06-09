"""
Minimise: abs(x) - 2*sqrt(y)
Subject to: e^x <= 2
            x+y = 5
x,y in R

https://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/exercises/hello_world.ipynb

Notes:
Convex problems require the objective function and all inequality constraints to be convex, and all equality constraints to be linear.
e^x is a convex function (need all inequality constraints convex), as its 2nd deriv is >0 for all x (e^x > 0 for all x in R)
x+y = 5 is clearly linear
f(x,y) = abs(x) - 2*sqrt(y) has Hessian (matrix of 2nd derivs): [[0, 0], [0, (1/2)y^-(3/2)]]
In multiple dimension, functions are convex if Hessian is positive (semi-)definite - i.e. (z^T)Hz >= 0
For vector z = (z1, z2)^T we have, for function above, (z^T)Hz = {(z2)^2}/2 * y^-(3/2) which is >=0 as sqrt(y) can't be negative 
Hence, the problem is convex
"""
import numpy as np
import cvxpy as cp
from math import sqrt, exp

# Construct the problem.
x = cp.Variable(1)
y = cp.Variable(1)

# Form the objective expression, and check its curvature and sign:
expression = cp.abs(x) - 2*cp.sqrt(y)
print("Expression Curvature/Sign/is dcp?:", expression.curvature, expression.sign, expression.is_dcp())

# Set-up the problem
objective = cp.Minimize(expression)
constraints = [cp.exp(x) <= 2.0, x + y == 5.0]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()

# Optimal x,y and objective:
x_opt, y_opt = (x.value[0], y.value[0])
print("(x,y):", (x_opt, y_opt))
print("status:", prob.status)
print("optimal value", prob.value, abs(x_opt) - 2*sqrt(y_opt))

# Check constraints:
print("Check Constraints")
print("e^x <= 2:", exp(x_opt))
print("x+y=5:", x_opt+y_opt)

