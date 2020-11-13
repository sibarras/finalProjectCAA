import numpy as np
import scipy as sp
"""Maximize Function with linear Solving System

objective: z = 3*x1 + 5*x2

subject to:
x1 <= 4
2*x2 <= 12
3*x1 + 2*x2 <= 18
x1 >= 0
x2 >= 0

"""
c = [-3, -5]  # Funcion objetivo. Se usa menos porque se utiliza en el linprog funcion minimizar
A = [[1, 0], [0, 2], [3, 2]]  # Matriz de variables
b = [4, 12, 18]  # Matriz de limites
x0_bounds = (0, None)  # Limites de variable 1
x1_bounds = (0, None)  # limites de variable 2

from scipy.optimize import linprog
# Solve the problem by Simplex method in Optimization
res = linprog(c, A_ub=A, b_ub=b,  bounds=(x0_bounds, x1_bounds), method='simplex', options={"disp": True})

