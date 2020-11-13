import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(x:float):
    y = (x - 1.5)**2 + 0.5
    print("x={}, y={}".format(x,y))
    return y

xguess = 2.0
result = spo.minimize(f, xguess, method='SLSQP', options={'disp':True})

print("Minimum found at:\nRESULT: x={}, y={}".format(result.x, result.fun))