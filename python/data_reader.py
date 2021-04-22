from typing import NamedTuple
import pandas as pd
from pathlib import Path
from typing import NamedTuple

class CNDdata(NamedTuple):
    capacity: pd.DataFrame
    restrictions: pd.DataFrame
    termos: pd.DataFrame
    hidros: pd.DataFrame

def load_cnd_data(file: Path) -> CNDdata:
    """The function geneates a tuple of dataframes, with information about the data extracted from file.

    Args:
        file (Path): The file directory with the information in separate sheets.

    Returns:
        CNDdata: Returns a CNDdata namedtuple object with Capacity, restrictions, termos and hidros dataframes.
    """

    # Leemos el excel
    wb: dict[str, pd.DataFrame] = pd.read_excel(file, None)

    # Arreglamos los df que no tienen titulo en su columna ## AQUI FALTA ARREGLAR QUE QUEDA INDICE 1 DE INICIO EN EL ARREGLADO
    # Esta funcion no es general, solo agarra la primera fila y la coloca como fila de titulo. Mejorar
    ordered_df = lambda df : (df.iloc[1:].rename(columns=df.iloc[0]) if df.columns.str.contains('^Unnamed').any() else df)
    wb_list:list[pd.DataFrame] = [ordered_df(df) for df in wb.values()]

    # Arreglamos los nombres de las columnas para eliminar los espacios
    wb_list = [df.rename(columns={name:name.strip() for name in df.columns}) for df in wb_list]

    # Eliminamos columnas duplicadas
    cnd_data = CNDdata(*[df.loc[:, ~df.columns.duplicated()] for df in wb_list])

    return cnd_data


def load_demand(filename:Path) -> pd.Series:
    """Loads from excel file given, the maximum demand per hour for one complete week. Returns a pd.Series object with the numbers.

    Args:
        filename (Path): The excel directory with the information.

    Returns:
        pd.Series: Returns the information in a pd.Series.
    """
    
    # Cargar la demanda eléctrica
    demand_df:pd.DataFrame = pd.read_excel(filename)

    # Corregir el espaciado en las columnas
    demand_df = demand_df.rename(columns={col: str(col).strip() for col in demand_df.columns})

    # Obtener el vector de Demanda
    demand_data = demand_df.loc[3:, 'MW']

    return demand_data


if __name__ == '__main__':
    main_path = Path().parent
    excel_data_path = main_path / 'excel'

    cnd_data_file = excel_data_path/'cnd_data.xlsx'
    demand_file = excel_data_path/'demanda.xlsx'

    cnd_data = load_cnd_data(cnd_data_file)
    demand_data = load_demand(demand_file)



