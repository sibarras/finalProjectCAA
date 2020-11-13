import numpy as np
import pandas as pd
from scipy.optimize import linprog

enunciado = """
La compañía de luz tiene tres centrales que cubren las necesidades de cuatro ciudades. Cada central
suministra las cantidades siguientes de kilowatts-hora: planta 1, 35 millones; planta 2, 50 millones; planta 3, 40 millones. Las demandas de potencia pico en estas ciudades que ocurren a la misma hora (2:00 p.m.) son como sigue (en kw/h): ciudad 1, 45 millones; ciudad 2, 20 millones; ciudad 3, 30 millones y
ciudad 4, 30 millones. Los costos por enviar un millón de kw/h de la planta dependen de la distancia
que debe viajar la electricidad y se muestran en la tabla A.
"""
c1, c2, c3, c4 = 'ciudad 1', 'ciudad 2', 'ciudad 3', 'ciudad 4'
cols = [c1, c2, c3, c4]

p1, p2, p3 = 'planta 1', 'planta 2', 'planta 3'
idx = [p1, p2, p3]

plantas = [
    [8, 6, 10, 9, 35],
    [9, 12, 13, 7, 50],
    [14, 20, 16, 5, 40],
    [45, 20, 30, 40, None]
]

tabla_costos_generacion = pd.DataFrame(plantas, index=idx[:].append('demanda'), columns=cols[:].append('generacion'))
matriz_variables = tabla_costos_generacion.to_numpy(dtype=float)[:-1, :-1]

# Matriz de objetivo a optimizar
matriz_objetivo = matriz_variables.flatten()

# Para hacer la matriz de restricciones
zeros = np.zeros_like(matriz_variables)

# Matriz de restriccion
matriz_restriccion_variables = np.array([
    np.append(np.array([]), [np.ones(4), np.zeros(4), np.zeros(4)]),
    np.append(np.array([]), [np.zeros(4), np.ones(4), np.zeros(4)]),
    np.append(np.array([]), [np.zeros(4), np.zeros(4), np.ones(4)]),
    -np.insert(np.delete(np.zeros_like(matriz_variables), 0, 1), 0, np.ones(3), axis=1).flatten(),
    -np.insert(np.delete(np.zeros_like(matriz_variables), 1, 1), 1, np.ones(3), axis=1).flatten(),
    -np.insert(np.delete(np.zeros_like(matriz_variables), 2, 1), 2, np.ones(3), axis=1).flatten(),
    -np.insert(np.delete(np.zeros_like(matriz_variables), 3, 1), 3, np.ones(3), axis=1).flatten()
])

# Limites de cada restriccion (Todas descritas como variables mayor o igual que constantes)
limites = np.array([35,50,40, -45, -20, -30, -30])

resultado = linprog(matriz_objetivo, matriz_restriccion_variables, limites, method='simplex')

print(f'\nResultado:\n\nValor de variables:\n{resultado.x}\n\nValor Optimizado: {resultado.fun} usd')

tabla_de_generacion = pd.DataFrame(resultado.x.reshape(3, 4), index=idx, columns=cols)
print('\n\n',tabla_de_generacion)
