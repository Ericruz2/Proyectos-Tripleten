# Sprint 14: Rusty Bargain: Modelado Predictivo Eficiente para Precios de Vehículos
# El servicio de venta de autos usados Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes. 
# Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial:
# especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# A Rusty Bargain le interesa:

# - La calidad de la predicción;
# - La velocidad de la predicción;
# - El tiempo requerido para el entrenamiento

## Preparación de datos

import time

# Librerías para manipulación de datos
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
import lightgbm as lgb

# Cargar dataset
df = pd.read_csv('/datasets/car_data.csv')

# Información general
print(df.info())
print("Valores faltantes por columna:\n", df.isna().sum())

# Conteo combinado de faltantes y atípicos
conteo_invalidos = {
    'Price = 0': (df['Price'] == 0).sum(),
    'Power = 0': (df['Power'] == 0).sum(),
    'VehicleType missing': df['VehicleType'].isna().sum(),
    'Gearbox missing': df['Gearbox'].isna().sum(),
    'Model missing': df['Model'].isna().sum(),
    'FuelType missing': df['FuelType'].isna().sum(),
    'NotRepaired missing': df['NotRepaired'].isna().sum()
}

print("🔍 Conteo de registros con valores faltantes o atípicos:\n")
for tipo, cantidad in conteo_invalidos.items():
    print(f"{tipo}: {cantidad}")

total = sum(conteo_invalidos.values())
print(f"\n🧮 Total acumulado de registros con problemas: {total}")

# Total de registros
total_registros = df.shape[0]
print(f"Total de registros: {total_registros:,}")

# Detectar todos los duplicados (incluye la primera aparición)
duplicados_totales = df[df.duplicated(keep=False)]
print(f"Total de registros duplicados detectados: {duplicados_totales.shape[0]:,}")

# Detectar solo los que serían eliminados (no incluye la primera aparición)
duplicados_marcados = df[df.duplicated()]
print(f"Registros marcados como duplicados: {duplicados_marcados.shape[0]:,}")

# Eliminar duplicados (conservando la primera aparición)
original_len = len(df)
df = df.drop_duplicates().reset_index(drop=True)
sin_duplicados_len = len(df)

# Visualización comparativa
valores = [original_len, sin_duplicados_len]
labels = ['Original', 'Sin duplicados']

plt.figure(figsize=(6, 4))
plt.bar(labels, valores, color=['coral', 'lightgreen'])
plt.title('Tamaño del dataset antes y después de eliminar duplicados')
plt.ylabel('Cantidad de registros')

# Mostrar valores encima de las barras
for i, val in enumerate(valores):
    plt.text(i, val + original_len * 0.01, f'{val:,}', ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Función para mostrar resumen estadístico con percentiles (para detectar outliers)
def resumen_outliers(data, columna):
    stats = data[columna].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    print(f"\nResumen de {columna} (datos sin duplicados):\n{stats.round(2)}")

# Aplicar a variables numéricas clave
columnas_clave = ['Price', 'Power', 'Mileage', 'RegistrationYear']
for col in columnas_clave:
    resumen_outliers(df, col)

# Filtrado técnico sobre variables clave
df_filtrado = df[
    (df['Price'] >= 350) & (df['Price'] <= 19000) &
    (df['Power'] >= 30) & (df['Power'] <= 306) &
    (df['Mileage'] >= 5000) &
    (df['RegistrationYear'] >= 1950) & (df['RegistrationYear'] <= 2020)
].copy()

# Resumen comparativo antes/después del filtrado técnico
original = len(df)
post_filtro = len(df_filtrado)
print(f"Registros antes del filtrado técnico: {original:,}")
print(f"Registros después del filtrado técnico: {post_filtro:,}")
print(f"Registros eliminados: {original - post_filtro:,} ({100 * (original - post_filtro)/original:.2f}%)")

#  Filtrado estadístico por IQR
def filtrar_iqr(data, columna):
    Q1 = data[columna].quantile(0.25)
    Q3 = data[columna].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    return data[(data[columna] >= lim_inf) & (data[columna] <= lim_sup)], lim_inf, lim_sup

impacto_outliers = []
df_filtrado_iqr = df_filtrado.copy()

#  Aplicación del filtrado IQR por columna
for col in ['Price', 'Power', 'Mileage', 'RegistrationYear']:
    df_col_filtrado, lim_inf, lim_sup = filtrar_iqr(df_filtrado_iqr, col)
    registros_antes = len(df_filtrado_iqr)
    registros_despues = len(df_col_filtrado)
    eliminados = registros_antes - registros_despues

    impacto_outliers.append({
        'Variable': col,
        'Límite Inferior': round(lim_inf, 2),
        'Límite Superior': round(lim_sup, 2),
        'Registros Eliminados': eliminados,
        '% Afectado': round(eliminados / registros_antes * 100, 2)
    })

    df_filtrado_iqr = df_col_filtrado.copy()

# Versión final limpia
df_filtrado_final = df_filtrado_iqr.copy()

#  Mostrar resumen de impacto
df_impacto = pd.DataFrame(impacto_outliers)
print("\n --- > Resumen del impacto por variable:\n")
print(df_impacto.to_string(index=False))

# Función para comparar boxplots antes y después del IQR
def comparar_boxplots(variable):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(x=df_filtrado[variable], ax=axs[0], color='lightcoral')
    axs[0].set_title(f'{variable} - Antes del IQR')

    sns.boxplot(x=df_filtrado_final[variable], ax=axs[1], color='lightgreen')
    axs[1].set_title(f'{variable} - Después del IQR')

    plt.tight_layout()
    plt.show()

# Comparar variables clave visualmente
for var in ['Price', 'Power', 'Mileage']:
    comparar_boxplots(var)

# Grafico de RegistrationYear despues de IQR
plt.figure(figsize=(8, 4))
sns.histplot(df_filtrado_final['RegistrationYear'], bins=40, kde=False, color='green')
plt.title('Distribución de RegistrationYear (después del IQR)')
plt.xlabel('Año de registro')
plt.ylabel('Frecuencia')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Eliminar columnas irrelevantes
cols_to_drop = ['DateCrawled', 'DateCreated', 'LastSeen', 'PostalCode', 'NumberOfPictures']
df.drop(columns=cols_to_drop, inplace=True)

# Variables categóricas a revisar
categorical_cols = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'NotRepaired']

# Total de filas
total_rows = len(df)

# Valores faltantes y porcentaje
print("\n Valores faltantes por variable categórica:")
for col in categorical_cols:
    missing = df[col].isna().sum()
    pct_missing = round((missing / total_rows) * 100, 2)
    print(f"{col:12s}: {missing:6d} ({pct_missing:5.2f}%)")

# Distribución de clases con nulos visibles
for col in categorical_cols:
    print(f"\n Distribución de {col} (top 10):")
    print(df[col].value_counts(dropna=False).head(10))

# Rellenar con 'unknown'
df[categorical_cols] = df[categorical_cols].fillna('unknown')

# Verificar si aún quedan valores nulos
print("Cantidad de nulos por columna:")
print(df.isnull().sum())

# Confirmar si el dataset está completamente limpio
if df.isnull().sum().sum() == 0:
    print("\n:) El dataset está completamente limpio. No hay valores nulos.")
else:
    print(" Aún hay valores nulos en el dataset. Revisa el resumen anterior.")
    
# Medir uso de memoria antes
memoria_antes = df.memory_usage(deep=True).sum() / 1024**2  # En MB
print(f"-- > Memoria antes de convertir: {memoria_antes:.2f} MB")

# Convertir columnas categóricas a tipo 'category'
categorical_cols = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Medir uso de memoria después
memoria_despues = df.memory_usage(deep=True).sum() / 1024**2  # En MB
print(f"--- > Memoria después de convertir: {memoria_despues:.2f} MB")

# Comparación
reduccion = memoria_antes - memoria_despues
print(f"--> Reducción total: {reduccion:.2f} MB")

# Definimos el umbral mínimo de frecuencia
threshold = 100

# Creamos un conteo previo para evitar múltiples cálculos
model_counts = df['Model'].value_counts()

# Reemplazamos los modelos con baja frecuencia por 'other'
df['Model'] = df['Model'].apply(lambda x: x if model_counts[x] >= threshold else 'other')    

# Separar features y target
features = df.drop('Price', axis=1)
target = df['Price']

# Dividir conjunto de datos
X_train, X_val, y_train, y_val = train_test_split(
    features, target, test_size=0.25, random_state=42
)

print("--> Tamaño del conjunto de entrenamiento:", X_train.shape)
print("--> Tamaño del conjunto de validación:", X_val.shape)

# Matriz de correlación entre variables numéricas
correlation_matrix = df.select_dtypes(include='number').corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(" Matriz de Correlación entre Variables Numéricas")
plt.tight_layout()
plt.show()

# ----------- Entrenamiento del modelo -------------
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_val_encoded = pd.get_dummies(X_val, drop_first=True)

X_train_encoded, X_val_encoded = X_train_encoded.align(
    X_val_encoded, join='left', axis=1, fill_value=0
)

#  Copiar datos para LightGBM y convertir categóricas
X_train_lgb = X_train.copy()
X_val_lgb = X_val.copy()

cat_features = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']
for col in cat_features:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_val_lgb[col] = X_val_lgb[col].astype('category')
    
resultados = []

def evaluar_modelo(nombre, modelo, X_tr, X_val, y_tr, y_val):
    inicio_entrenamiento = time.time()
    modelo.fit(X_tr, y_tr)
    tiempo_entrenamiento = time.time() - inicio_entrenamiento

    inicio_prediccion = time.time()
    y_pred = modelo.predict(X_val)
    tiempo_prediccion = time.time() - inicio_prediccion

    rmse = mean_squared_error(y_val, y_pred, squared=False)

    resultados.append({
        "Modelo": nombre,
        "RMSE": round(rmse, 2),
        "Entrenamiento (s)": round(tiempo_entrenamiento, 2),
        "Predicción (s)": round(tiempo_prediccion, 2)
    })

    print(f"{nombre} \nRMSE: {rmse:.2f} \nEntrenamiento: {tiempo_entrenamiento:.2f}s \nPredicción: {tiempo_prediccion:.2f}s")

# Diccionario de modelos
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(max_depth=10),
    "Bosque Aleatorio": RandomForestRegressor(n_estimators=100, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(num_leaves=31, learning_rate=0.1, n_estimators=100)
}        

# Regresión Lineal
evaluar_modelo("Regresión Lineal", modelos["Regresión Lineal"], X_train_encoded, X_val_encoded, y_train, y_val)

#Árbol de decisión
evaluar_modelo("Árbol de Decisión", modelos["Árbol de Decisión"], X_train_encoded, X_val_encoded, y_train, y_val)

#Bosque aleatorio
evaluar_modelo("Bosque Aleatorio", modelos["Bosque Aleatorio"], X_train_encoded, X_val_encoded, y_train, y_val)

# Gradient Boosting
evaluar_modelo("Gradient Boosting", modelos["Gradient Boosting"], X_train_encoded, X_val_encoded, y_train, y_val)
# XGBoost
evaluar_modelo("XGBoost", modelos["XGBoost"], X_train_encoded, X_val_encoded, y_train, y_val)

# LightGBM con variables categóricas
evaluar_modelo("LightGBM", modelos["LightGBM"], X_train_lgb, X_val_lgb, y_train, y_val)


#----------------- Analisis del modelo -----------------
df_resultados = pd.DataFrame(resultados).sort_values(by="RMSE")
print("\n📋 Resultados finales:")
print(df_resultados)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_resultados,
    x="Entrenamiento (s)",
    y="RMSE",
    size="Predicción (s)",
    hue="Modelo",
    sizes=(100, 800),
    legend=False,
    alpha=0.7
)

# Agregar etiquetas a cada punto
for i in range(len(df_resultados)):
    plt.text(
        df_resultados["Entrenamiento (s)"].iloc[i] + 1,
        df_resultados["RMSE"].iloc[i],
        df_resultados["Modelo"].iloc[i],
        fontsize=9
    )

plt.title("Comparativa de modelos: Precisión vs Tiempos")
plt.xlabel("Tiempo de entrenamiento (s)")
plt.ylabel("RMSE")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Crear DataFrame para coche nuevo
nuevo_coche = pd.DataFrame([{
    'VehicleType': 'limousine',
    'RegistrationYear': 2020,
    'Gearbox': 'automatic',
    'Power': 110,
    'Model': 'golf',
    'Mileage': 35000,
    'RegistrationMonth': 6,
    'FuelType': 'petrol',
    'Brand': 'volkswagen',
    'NotRepaired': 'no'
}], columns=X_train.columns)

# Convertir las columnas categóricas
for col in cat_features:
    nuevo_coche[col] = nuevo_coche[col].astype('category')

# Verificar que coincidan las columnas
assert list(nuevo_coche.columns) == list(X_train.columns), "Las columnas no coinciden con el modelo entrenado"

# Predicción
prediccion = modelo_lightgbm.predict(nuevo_coche)
print(f"Precio estimado del vehículo: €{int(prediccion[0])}")