import pandas as pd
import numpy as np
from scipy.optimize import minimize
from obtencion_de_datos import dataframes, demand
from matplotlib import pyplot as plt

df_capacidad, df_restricciones, df_termicas, df_hidros = dataframes

assert type(df_capacidad) is pd.DataFrame and type(df_restricciones) is pd.DataFrame
assert type(df_termicas) is pd.DataFrame and type(df_hidros) is pd.DataFrame

# Para que el editor entienda el tipo de la variable
df_capacidad:pd.DataFrame
df_restricciones:pd.DataFrame
df_termicas:pd.DataFrame
df_hidros:pd.DataFrame

# Eliminar plantas termicas que no tengan costo de arranque
df_termicas = df_termicas.loc[df_termicas['.CVaria']!=0]

# funciones limites dependientes de las variables
demand:np.ndarray
vector_demandas = demand['PANAMA'].to_numpy()
demanda_actual = vector_demandas[0]

# Variables para la funcion de costo
cant_plantas = len(df_termicas) + len(df_hidros)
mw_instalado_hidros = df_hidros['....Pot'].to_numpy()

coste_por_mw = np.append(df_termicas['.CVaria'].to_numpy(), np.zeros(len(mw_instalado_hidros)))
coste_por_mw[len(df_termicas)+df_hidros.loc[df_hidros['Generadoras']=='Bayano'].index] = 50

coste_arranque = df_termicas['Cold Startup Cost'].replace(np.nan, 0).to_numpy()
coste_arranque = np.append(coste_arranque, np.zeros(len(mw_instalado_hidros)))


# Las variables seran los tiempos de generacion de todas las plantas y los mw de las termicas
cant_variables = 2*len(df_termicas)+len(df_hidros)

# Variables de los limites
minimo_generacion = df_termicas['.Gen Min'].to_numpy().reshape(len(df_termicas),1)
maximo_generacion = df_termicas['.Gen Max'].to_numpy().reshape(len(df_termicas),1)

# Restriccion por colocar
# ratio_subida = df_restricciones['Max Rampa Subida (Mw/min)'].to_numpy()

# Las funciones que hagas pueden ser interactivas.

# x0 = [np.random.rand() for i in range(cant_plantas)]
# x0 = np.append(x0, np.random.randint(2,100, cant_variables-cant_plantas))
# print(x0)



# ================= UTILIZANDO EL OPTIMIZADOR ==========================

def optimizar_hora(hora:int, x0:np.ndarray):
    """ OPTIMIZA LA HORA INDICADA DENTRO DEL ARREGLO DE HORAS INTRODUCIDO """
    assert hora <= len(vector_demandas)
    demanda_actual = vector_demandas[hora]
    
    # Funcion de Costo
    def funcion_de_costo(x:np.ndarray):
        mw_operativos = np.append(x[cant_plantas:], mw_instalado_hidros)
        entrada_plantas = x[:cant_plantas]
        vector_aportes = mw_operativos * coste_por_mw * entrada_plantas
        resultado = vector_aportes @ np.ones_like(vector_aportes).T
        resultado += coste_arranque @ [(1 if h > 0.001 else 0) for h in vector_aportes]
        return resultado

    # Restricciones
    def generacion_inferior(x):
        mw_operativos = np.append(x[cant_plantas:], mw_instalado_hidros)
        res = (mw_operativos  @ x[:cant_plantas].T) - demanda_actual
        return res
    lim_generacion_inferior = {'type': 'eq', 'fun':generacion_inferior}

    def generacion_superior(x):
        mw_operativos = np.append(x[cant_plantas:], mw_instalado_hidros)
        res = demanda_actual*1.05 - (mw_operativos  @ x[:cant_plantas].T)
        return res
    lim_generacion_superior = {'type': 'ineq', 'fun':generacion_superior}

    # Para lograr que los valores sean enteros
    entero = {'type':'ineq','fun': lambda x : max([x[i]-int(x[i]) for i in range(len(x[:cant_plantas]))])}
    funciones_limites = [lim_generacion_inferior, lim_generacion_superior, entero]
    # for i in range(cant_plantas):
    #     # Mientras mas cerca de uno mejor
    #     fun = lambda x: 1-x[i]
    #     current = {'type': 'ineq', 'fun':fun}
    #     fun2 = lambda x: x[i]
    #     current2 = {'type': 'ineq', 'fun':fun2}
    #     funciones_limites.append(current)
    #     funciones_limites.append(current2)

    # for i in range(cant_variables-cant_plantas):
    #     fun = lambda x: maximo_generacion.flatten()[i]-x[i+cant_plantas]
    #     current = {'type': 'ineq', 'fun':fun}
    #     fun2 = lambda x: x[i+cant_plantas]-minimo_generacion.flatten()[i]
    #     current2 = {'type': 'ineq', 'fun':fun2}
    #     funciones_limites.append(current)
    #     funciones_limites.append(current2)


    #hess = lambda x, v: np.zeros((cant_variables, cant_variables))
    
    # Limites de Variables
    limites = np.append(minimo_generacion, maximo_generacion, axis=1)
    limites = np.vstack((np.array([[0, 1] for _ in range(cant_plantas)]), limites))

    # Parametros iniciales
    #print('\nParametros Iniciales:', x0)

    # Solucion del problema
    solucion = minimize(funcion_de_costo, x0, method='SLSQP', bounds=limites, constraints=funciones_limites)

    respuesta = f"""
    Para satisfacer la demanda de {demanda_actual} MW de la hora {vector_demandas.tolist().index(demanda_actual)}:

    Costo Minimo Encontrado: {solucion.fun}
    generacion alcanzada: {generacion_inferior(solucion.x) + demanda_actual}

    """
    print(respuesta)
    return solucion

def formatAnswer(respuesta:np.ndarray):
    results = {}
    tiempo_termicas, tiempo_hidros, pot_termicas = np.split(respuesta.x, [cant_plantas-len(mw_instalado_hidros), cant_plantas])
    nombres = np.append(df_termicas['Generadoras'].values, df_hidros['Generadoras'].values)
    n_termicas = len(tiempo_termicas)
    for i in range(len(nombres)):
        if i < n_termicas:
            results[nombres[i]] = (round(tiempo_termicas[i], 7), round(pot_termicas[i], 7), round(tiempo_termicas[i]*pot_termicas[i], 7))
            results[nombres[i]] += (round(results[nombres[i]][-1]*coste_por_mw[i] + coste_arranque[i]*(1 if results[nombres[i]][-1] > 0.001 else 0), 7),)
        else:
            results[nombres[i]] = (round(tiempo_hidros[i-n_termicas], 7), round(mw_instalado_hidros[i-n_termicas], 7),\
                                round(tiempo_hidros[i-n_termicas]*mw_instalado_hidros[i-n_termicas], 7))
            results[nombres[i]] += (round(results[nombres[i]][-1]*coste_por_mw[i] + coste_arranque[i]*(1 if results[nombres[i]][-1] > 0.001 else 0), 7),)
    
    totales = [0 for _ in range(len(list(results.values())[0]))]
    for val1, val2, val3, val4 in results.values():
        totales[0]+=val1
        totales[1]+=val2
        totales[2]+=val3
        totales[3]+=val4
    results['Total'] = totales

    df_solution = pd.DataFrame.from_dict(results, orient='index', columns=['Horas en Generacion', 'Potencia Generada', 'Aporte', 'Costo'])
    print('Tabla de resultados para la hora {}:'.format(hora))
    print(df_solution)
    return df_solution

def convertir_a_excel(semana):
    print('\n\nCreando Libros de Excel...')
    #from openpyxl import Workbook
    for num, df in enumerate(semana):
        if num == 0:
            dia = [df]
        dia.append(df)

        if (num+1) % 24 == 0:
            str_dia = './excel/results/dia_{}.xlsx'.format((num+1)//24)
            writer = pd.ExcelWriter(str_dia)
            for hora, datos_hora in enumerate(dia):
                datos_hora.to_excel(writer, 'hora_{}'.format(hora), columns=datos_hora.columns.values)
            writer.save()
            dia = []
    print('\nLibros Creados!')

def graficar_resultados(semana:list):
    pass



if __name__ == "__main__":
    semana = []
    for hora in range(len(vector_demandas)):
        x0 = (np.ones(cant_variables)*0.99 if 'soluc' not in locals() else soluc.x)


        x0 = np.ones(cant_variables)*0.99
        soluc = optimizar_hora(hora, x0)
        df = formatAnswer(soluc)
        semana.append(df)

    convertir_a_excel(semana)
    graficar_resultados(semana)
