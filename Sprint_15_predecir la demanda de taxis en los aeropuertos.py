import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display

df = pd.read_csv ('C:/Users/agustin/Downloads/taxi.csv',  parse_dates=['datetime'], index_col='datetime')

# Vista general
print (df.info())
display(df.head())

# Copia para limpieza
df_clean = df.copy()

# Verificación de duplicados
duplicados_total = df_clean.duplicated().sum()
print(f"🔍 Filas duplicadas: {duplicados_total}")

# Verificación de duplicados en el índice
duplicados_index = df_clean.index.duplicated().sum()
print(f"🕒 Timestamps duplicados: {duplicados_index}")

# Eliminación segura
df_clean = df_clean.drop_duplicates()

# Filtrar y mostrar filas con valores NaN en 'num_orders'
nan_rows = df[df['num_orders'].isna()]
print("🕳️ Registros con valores NaN en 'num_orders':")
print(nan_rows)

nan_rows = df_hourly[df_hourly['num_orders'].isna()]

# Filtrar filas con al menos un NaN
nan_rows = df[df.isna().any(axis=1)]

# Mostrar las columnas con NaN en cada fila
print("🔍 Valores NaN por fila:")
print(nan_rows)

# Mostrar las columnas con NaN en cada fila (como lista)
df_nan_columns = df[df.isna().any(axis=1)].apply(lambda row: row[row.isna()].index.tolist(), axis=1)

print("📋 Columnas con NaN por fila:")
print(df_nan_columns)