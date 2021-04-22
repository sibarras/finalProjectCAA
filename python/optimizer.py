import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeResult
from data_reader import CNDdata, load_cnd_data, load_demand
from pathlib import Path

main_path = Path().parent
excel_data_path = main_path / 'excel'

cnd_data_file = excel_data_path/'cnd_data.xlsx'
demand_file = excel_data_path/'demanda.xlsx'
results_path = excel_data_path/'new_results'

cnd_data = load_cnd_data(cnd_data_file)
demand_data = load_demand(demand_file)

df_capacidad, df_restricciones, df_termicas, df_hidros = cnd_data


# ================= Ajustar los datos obtenidos de los Archivos Excel ==========================

# Eliminar plantas termicas que no tengan costo de generacion, porque no estan en servicio
df_termicas = df_termicas.loc[df_termicas['.CVaria']!=0]

# =================================== NOMBRES DE VARIABLES ======================================

# Variables a optimizar:
# xi -> Tiempo de Generación de la térmica i en horas
# yj -> Tiempo de Generación de la hidroeléctrica j en horas
# zi -> Potencia de Generación de la térmica i en MW

# Constantes de Funcion de Optimizacion
# Ai -> Costo de generacion del MW en la termica i
# Bj -> Costo de generacion del MW en la hidro j
# Cj -> MW disponibles en la planta hidro i
# Di -> Costo de Arranque de la planta termica i
# Ei -> Costo de Parada de la planta termica i
# Fj -> Costo de Arranque de la central hidro j
# Gj -> Costo de Parada de la central hidro j

# Constantes de limitacion:
# Hi -> Limite minimo de generacion de la planta termica i
# Ii -> Limite maximo de generacion de la planta termica i
# Ji -> Limite minimo de tiempo de la central termica i
# Ki -> Limite maximo de tiempo de la central termica i
# Lj -> Limite minimo de tiempo de la central hidro j
# Mj -> Limite maximo de tiempo de la central hidro j
# Nk -> Demanda energetica en la hora k elegida

# ================= Obtenemos las variables necesarias para el optimizador ====================

# Cantidad de Termicas, Hidros
n, m = len(df_termicas), len(df_hidros)

# Variables a Optimizar
x:np.ndarray = np.ones(n)
y:np.ndarray = np.ones(m)
z:np.ndarray = np.ones(n)

# Reunimos todas las variables del problema en el vector vars
vars:np.ndarray = np.concatenate((x, y, z), axis=None)*0.99
vars_len = len(vars)

# Constantes
A = df_termicas['.CVaria'].to_numpy()
B = np.array([(50 if i == df_hidros.loc[df_hidros['Generadoras']=='Bayano'].index else 0) for i in range(m)])
C = df_hidros['....Pot'].to_numpy()
D = df_termicas['Cold Startup Cost'].replace(to_replace=np.nan, value=0).to_numpy()
E = np.zeros_like(x) # E no esta considerado
F = np.zeros_like(y)
G = np.zeros_like(y) # G no esta considerado

# Const de restricciones
H = df_termicas['.Gen Min'].to_numpy()
I = df_termicas['.Gen Max'].to_numpy()
J = np.zeros_like(x)
K = np.ones_like(x)
L = np.zeros_like(y)
M = np.ones_like(y)
N = demand_data.to_numpy()

CONST = A,B,C,D,E,F,G,H,I,J,K,L,M,N

# ===================================== FUNCION DE COSTO ========================================

# Llamaremos a la funcion de costo:
# f(x1, ..., xi, ..., xn,
#   y1, ..., yj, ..., ym,
#   z1, ..., zi, ..., zn) = Sum |i=1 -> n| {Ai*xi*zi + Di+Ei} + Sum |j=1 -> m| {Bj*yj*Cj + Fj+Gj} 

# Funciones Complementarias
def energy(time:np.ndarray, pot:np.ndarray) -> np.ndarray:
    return time*pot

def cost(mw_cost:np.ndarray, energy:np.ndarray, const:np.ndarray) -> np.ndarray:
    return mw_cost*energy + const

# Funcion de Costo
def f(vars: np.ndarray) -> float:
    x, y, z = vars[:n], vars[n:n+m], vars[n+m:]
    termos_energy, hidros_energy = energy(x, z), energy(y, C)
    termos_cost, hidros_cost = cost(A, termos_energy, D), cost(B, hidros_energy, F)
    return sum(termos_cost) + sum(hidros_cost)


# ============================= Limites del problema =====================================

# Minimo y Maximo de Generacion para cumplir la demanda

# Sum |i=1 -> n| {xi*zi} + Sum |j=1 -> m| yj*Cj > Nk
def g1(nk:float):
    def g11(vars: np.ndarray) -> float:
        x, y, z = np.split(vars, [n, n+m])
        return sum(x*z + D) + sum(y*C + F) - nk
    return g11

# Sum |i=1 -> n| {xi*zi} + Sum |j=1 -> m| yj*Cj < 1.01*Nk
def g2(nk:float):
    def g22(vars: np.ndarray) -> float:
        x, y, z = vars[:n], vars[n:n+m], vars[n+m:]
        return 1.01*nk - sum(x*z + D) + sum(y*C + F)
    return g22


# Minimo y Maximo de Generacion de las termicas (zi)
# zi > Hi ^ zi < Ii
z_bounds = np.array([[min, max] for min, max in zip(H, I)])

# Minimo y Maximo de tiempo de generacion (xi, yi)
# xi > Ji ^ xi < Ki
# yj > Lj ^ yj < Mj
x_bounds = np.array([[min, max] for min, max in zip(J, K)])
y_bounds = np.array([[min, max] for min, max in zip(L, M)])

# Limites para el optimizador
get_constraints = lambda nk: [{'type': 'ineq', 'fun':fun} for fun in [g1(nk), g2(nk)]]
BOUNDS = np.vstack((x_bounds, y_bounds, z_bounds))


# ================= UTILIZANDO EL OPTIMIZADOR ==========================

def optimize(hour:int, x0:np.ndarray):
    nk = N[hour]
    solution:OptimizeResult = minimize(f, vars, method='SLSQP', constraints=get_constraints(nk), bounds=BOUNDS)

    respuesta = f"""
    Para satisfacer la demanda de {N[hour]} MW de la hora {hour}:

    Costo Minimo Encontrado: {solution.fun}
    generacion alcanzada: {g1(nk)(solution.x) + nk}

    """
    print(respuesta)
    return solution

# ================= CREANDO UN DATAFRAME CON LAS RESPUESTAS DE LA HORA ==========================

def format_answer(solution:OptimizeResult, cnd_data:CNDdata, const:tuple[np.ndarray, ...], hour:int):
    x_opt, y_opt, z_opt = np.split(solution.x, [n, n+m])
    df_termicas, df_hidros = cnd_data.termos, cnd_data.hidros
    A,B,C,D,E,F,G = const[:7]

    termo_names = df_termicas['Generadoras'].values
    hidro_names = df_hidros['Generadoras'].values
    # decimals:int = 7
    termos_energy, hidros_energy = energy(x_opt, z_opt), energy(y_opt, C)
    termos_cost, hidros_cost = cost(A, termos_energy, D), cost(B, hidros_energy, F)
    results = {name: (time, mw, mwh, cost) for name, time, mw, mwh, cost in zip(termo_names, x_opt, z_opt, termos_energy, termos_cost)}
    results |= {name: (time, mw, mwh, cost) for name, time, mw, mwh, cost in zip(hidro_names, y_opt, C, hidros_energy, hidros_cost)}

    tot_time:float = sum(x_opt) + sum(y_opt)
    tot_pow = sum(z_opt) + sum(C)
    tot_energy = sum(termos_energy) + sum(hidros_energy)
    tot_cost = sum(termos_cost) + sum(hidros_cost)
    results['Total'] = tot_time, tot_pow, tot_energy, tot_cost

    df_solution = pd.DataFrame.from_dict(results, orient='index', columns=['Horas en Generacion', 'Potencia Generada', 'Aporte', 'Costo'])
    print(f'Tabla de resultados para la hora {hour}:')
    print(df_solution)
    return df_solution

# ================= CONVIRTIENDO LOS DATOS ENCONTRADOS A LIBROS DE EXCEL ==========================

def weekdata_to_excel(week_df_array: np.ndarray, results_path:Path):

    week_dfs_per_day = np.split(week_df_array, 7)
    week_dfs_dict:dict[int, dict[int, pd.DataFrame]] = {d:{h: df for h, df in enumerate(day_df_list)} for d, day_df_list in enumerate(week_dfs_per_day, 1)}
    
    print('\n\nCreando Libros de Excel...')
    for day, day_dfs_dict in week_dfs_dict.items():
        file = results_path / f'day_{day}.xlsx'
        with pd.ExcelWriter(file) as wb:
            for hour, hour_df in day_dfs_dict.items():
                hour_df.to_excel(wb, f'hour_{hour}', columns = hour_df.columns.values)

    print('\nLibros Creados!')


if __name__ == "__main__":
    week_df_array = np.zeros(N.shape, dtype=pd.DataFrame)
    for hour in range(len(N)):
        soluc = optimize(hour, vars)
        df = format_answer(soluc, cnd_data, CONST, hour)
        week_df_array[hour] = df

    weekdata_to_excel(week_df_array, results_path)


