import numpy as np
import pandas as pd
from scipy.optimize import linprog
from excel_to_sqlite3 import new_data

df_capacidad, df_restricciones, df_termicas, df_hidros = new_data

cols = ['generacion', 'restriccion min.', 'restriccion max.', 'coste', 'mantenimiento']
idx = ['motor nacional', 'motor de taiwan', 'motor desconocido', 'demanda pico', 'operario']

data = [
    [5, 0.5, 5, 40000, 0.5],
    [9, 1, 8, 64000, 0.25],
    [12, 2, 12, 170000, 0],
    [None, 180, None, None, None],
    [None, None, 5, None, None]
]
df = pd.DataFrame(data, index=idx, columns=cols)

# Ahora que tenemos los datos, creamos la funcion de la variable a optimizar como una matriz
generadoras = df.loc[pd.notnull(df['generacion'])]
otras_restricciones = df.loc[pd.isnull(df['generacion'])]


vector_coste = [coste for coste in generadoras['coste']]
print('\nVector de coste:', vector_coste)

restricciones = np.array([])
lim_max = np.array([])

# Restricciones y Limites de generacion de cada planta.
# TIENE SOLO MINIMO (1 vez)
generacion_plantas = generadoras['generacion']
restricciones = np.append(restricciones, -generacion_plantas).reshape(1,len(generadoras))
# Constante limite
minimo_generacion = otras_restricciones['restriccion min.']['demanda pico']
lim_max = np.append(lim_max, -minimo_generacion)

# Restricciones y limites de horas de mantenimiento para cada planta por hora de funcionamiento
# TIENE SOLO MAXIMO (1 vez negativo)
#mantenimiento_plantas = np.identity(len(generadoras))*np.array(generadoras['mantenimiento'])
mantenimiento_plantas = np.array(generadoras['mantenimiento']).reshape(1, len(generadoras))
restricciones = np.append(restricciones, mantenimiento_plantas, axis=0)

max_mant = otras_restricciones['restriccion max.']['operario']
lim_max = np.append(lim_max, max_mant)


# Limites directos de las variables a generar
limites_de_variables = np.array([])

min_horas, max_horas = generadoras['restriccion min.'].to_numpy(), generadoras['restriccion max.'].to_numpy()
limites_de_variables = np.append(limites_de_variables, min_horas).reshape(len(generadoras), 1)
limites_de_variables = np.append(limites_de_variables, max_horas.reshape(len(generadoras), 1), axis=1)

assert len(lim_max)==len(restricciones)

print('\nRestricciones:')
print(restricciones, end='\n\n')
print('Restricciones Minimas:')
print(lim_max, end='\n\n')
print('Restricciones de Variables generadas (min, max):')
print(limites_de_variables, end='\n\n')

# ## Ahora, vamos a correr el programa para determinar cuales son los valores optimos.

solution = linprog(vector_coste, restricciones, lim_max, bounds=limites_de_variables, method='simplex', options={'disp': True})
print('Variables generadas para cada generadora:', solution.x)
print('Coste de cada una de las variables:', solution.x)
