import pandas as pd
import numpy as np
from scipy.optimize import minimize
from data_reader import load_cnd_data, load_demand
from pathlib import Path

main_path = Path().parent
excel_data_path = main_path / 'excel'

cnd_data_file = excel_data_path/'cnd_data.xlsx'
demand_file = excel_data_path/'demanda.xlsx'

cnd_data = load_cnd_data(cnd_data_file)
demand_data = load_demand(demand_file)

df_capacidad, df_restricciones, df_termicas, df_hidros = cnd_data


# ================= Ajustar los datos obtenidos de los Archivos Excel ==========================

# Eliminar plantas termicas que no tengan costo de arranque
df_termicas = df_termicas.loc[df_termicas['.CVaria']!=0]

# funciones limites dependientes de las variables
vector_demandas:np.ndarray = demand_data.to_numpy()#*1.5

# ================= Obtenemos las variables necesarias para el optimizador ====================

cant_plantas = len(df_termicas) + len(df_hidros)


mw_disponible_hidros = df_hidros['....Pot'].to_numpy()


# Se define el costo de hidros igual a cero, excepto Bayano, con un coste de 50 por ser embalse
coste_por_mw = np.append(df_termicas['.CVaria'].to_numpy(), np.zeros(len(mw_disponible_hidros)))
coste_por_mw[len(df_termicas)+df_hidros.loc[df_hidros['Generadoras']=='Bayano'].index] = 50


# Se considera el coste de arranque de las centrales termicas a utilizar
coste_arranque = df_termicas['Cold Startup Cost'].replace(to_replace=np.nan, value=0).to_numpy()
coste_arranque = np.append(coste_arranque, np.zeros(len(mw_disponible_hidros)))


# Las variables seran los tiempos de generacion de todas las plantas y los mw de las termicas
cant_variables = 2*len(df_termicas)+len(df_hidros)


# Variables de los limites
minimo_generacion = df_termicas['.Gen Min'].to_numpy().reshape(len(df_termicas),1)
maximo_generacion = df_termicas['.Gen Max'].to_numpy().reshape(len(df_termicas),1)



# ================= UTILIZANDO EL OPTIMIZADOR ==========================

def optimizar_hora(hora:int, x0:np.ndarray):
    """ OPTIMIZA LA HORA INDICADA DENTRO DEL ARREGLO DE HORAS INTRODUCIDO """
    assert hora <= len(vector_demandas)
    demanda_actual = vector_demandas[hora]
    
    # Funcion de Costo
    def funcion_de_costo(x:np.ndarray):
        mw_operativos = np.append(x[cant_plantas:], mw_disponible_hidros)
        entrada_plantas = x[:cant_plantas]
        vector_aportes = mw_operativos * coste_por_mw * entrada_plantas
        resultado = vector_aportes @ np.ones_like(vector_aportes).T
        resultado += coste_arranque @ [(1 if h > 0.001 else 0) for h in vector_aportes]
        return resultado

    # Restricciones
    def generacion_inferior(x):
        mw_operativos = np.append(x[cant_plantas:], mw_disponible_hidros)
        res = (mw_operativos  @ x[:cant_plantas].T) - demanda_actual
        return res
    lim_generacion_inferior = {'type': 'eq', 'fun':generacion_inferior}

    def generacion_superior(x):
        mw_operativos = np.append(x[cant_plantas:], mw_disponible_hidros)
        res = demanda_actual*1.01 - (mw_operativos  @ x[:cant_plantas].T)
        return res
    lim_generacion_superior = {'type': 'ineq', 'fun':generacion_superior}

    # Para lograr que los valores sean enteros
    entero = {'type':'ineq','fun': lambda x : max([x[i]-int(x[i]) for i in range(len(x[:cant_plantas]))])}
    funciones_limites = [lim_generacion_inferior, lim_generacion_superior, entero]

    # Limites de Variables
    limites = np.append(minimo_generacion, maximo_generacion, axis=1)
    limites = np.vstack((np.array([[0, 1] for _ in range(cant_plantas)]), limites))

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
    respuesta = np.split(respuesta.x, [cant_plantas-len(mw_disponible_hidros), cant_plantas])
    tiempo_termicas, tiempo_hidros, pot_termicas = respuesta
    nombres = np.append(df_termicas['Generadoras'].values, df_hidros['Generadoras'].values)
    n_termicas = len(tiempo_termicas)
    for i in range(len(nombres)):
        if i < n_termicas:
            results[nombres[i]] = (round(tiempo_termicas[i], 7), round(pot_termicas[i], 7), round(tiempo_termicas[i]*pot_termicas[i], 7))
            results[nombres[i]] += (round(results[nombres[i]][-1]*coste_por_mw[i] + coste_arranque[i]*(1 if results[nombres[i]][-1] > 0.001 else 0), 7),)
        else:
            results[nombres[i]] = (round(tiempo_hidros[i-n_termicas], 7), round(mw_disponible_hidros[i-n_termicas], 7),\
                                round(tiempo_hidros[i-n_termicas]*mw_disponible_hidros[i-n_termicas], 7))
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
            dia = []
        dia.append(df)

        if (num+1) % 24 == 0:
            str_dia = './excel/results/dia_{}.xlsx'.format((num+1)//24)
            writer = pd.ExcelWriter(str_dia)
            for hora, datos_hora in enumerate(dia):
                datos_hora.to_excel(writer, 'hora_{}'.format(hora), columns=datos_hora.columns.values)
            writer.save()
            dia = []
    print('\nLibros Creados!')


if __name__ == "__main__":
    semana = []
    for hora in range(len(vector_demandas)):
        x0 = np.ones(cant_variables)*0.99
        soluc = optimizar_hora(hora, x0)
        df = formatAnswer(soluc)
        semana.append(df)

    convertir_a_excel(semana)


