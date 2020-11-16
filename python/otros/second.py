import numpy as np
from scipy.optimize import minimize


def objective(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return x1 * x4 * (x1 + x2 + x3) + x3


def constraint1(x):
    return x[0] * x[1] * x[2] * x[3] - 25


def constraint2(x):
    y = 40
    for n in x:
        y -= n ** 2
    return y


x0 = np.zeros(4)

b = (1, 5)
bnds = (b, b, b, b)

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
const = [con1, con2]

x0 = [1,1,1,1]

solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=const, options={'disp': True})


print(f'y={solution.fun}, x={solution.x}')

print('\n\n\n\n\n\nPRUEBA\n\n\n\n\n')

print(round(0.44))
print(round(0.77))
print(float(0.65547))
num = 0.76554
