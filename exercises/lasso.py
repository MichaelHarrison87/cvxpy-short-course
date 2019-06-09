"""
LASSO - Least-Squares Regression, with L1 Regularisation


https://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/exercises/Lasso.ipynb
"""

import numpy as np
import cvxpy as cp

n = 200
x = cp.Variable((n, 1))