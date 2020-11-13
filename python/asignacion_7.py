import numpy as np
import pandas as pd
from scipy.optimize import linprog

cols = ['generacion', 'restriccion min.', 'restriccion max.', 'coste', 'mantenimiento']
idx = ['motor nacional', 'motor de taiwan', 'motor desconocido', 'demanda pico', 'operario']

data = [
    [5, 0.5, 5, 40000, 0.5],
    [9, 1, 8, 64000, 0.25],
    [12, 2, 12, 17000, 0],
    [None, 180, None, None, None],
    [None, None, 5, None, None]
]

df = pd.DataFrame(data, index=idx, columns=cols)
# print(df, '\n\n')


