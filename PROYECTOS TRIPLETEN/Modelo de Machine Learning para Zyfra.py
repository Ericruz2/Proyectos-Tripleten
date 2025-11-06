# -------------- Introducción ----------------
# Este proyecto de ciencia de datos tiene como objetivo desarrollar un modelo de machine learning para Zyfra,
# una empresa que busca optimizar la eficiencia en la industria pesada.Para maximizar la rentabilidad, es crucial optimizar cada etapa del procesamiento del mineral. 
# Este proyecto se enfoca en el análisis y modelado predictivo del proceso de recuperación de oro en una planta de flotación. 
# Se trabaja con datos históricos del proceso para desarrollar un modelo que prediga con precisión el rendimiento de recuperación del oro.
#Este enfoque basado en datos permitirá mejorar la eficiencia de los procesos mineros, reduciendo pérdidas y optimizando la recuperación de oro.

# --------------- Objetivo -----------------
# Desarrollar un modelo predictivo que estime dos métricas clave del proceso de flotación:
# - rougher.output.recovery:
#   recuperación de oro en la etapa rougher

# - final.output.recovery:
#   recuperación de oro en la etapa final.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer

from IPython.display import Markdown

# Cargar los archivos
train = pd.read_csv('C:/Users/agustin/Downloads/gold_recovery_train.csv', index_col='date', parse_dates=True)
test = pd.read_csv('C:/Users/agustin/Downloads/gold_recovery_test.csv', index_col='date', parse_dates=True)
full = pd.read_csv('C:/Users/agustin/Downloads/gold_recovery_full.csv', index_col='date', parse_dates=True)

# Ver información general
print("Entrenamiento:")
print(train.info())
print("\nPrueba:")
print(test.info())
print("\nCompleto:")
print(full.info())

# ------------------- 1.2. Verificar el cálculo de rougher.output.recovery y encontrar a EAM -------------------
def calculate_recovery(C, F, T):
    return (C * (F - T)) / (F * (C - T)) * 100

# Columnas necesarias
cols_needed = [
    'rougher.output.concentrate_au',
    'rougher.input.feed_au',
    'rougher.output.tail_au',
    'rougher.output.recovery'
]

# Eliminar filas con datos faltantes
recovery_data = train[cols_needed].dropna()

# Calcular recuperación
recovery_calc = calculate_recovery(
    recovery_data['rougher.output.concentrate_au'],
    recovery_data['rougher.input.feed_au'],
    recovery_data['rougher.output.tail_au']
)

# Valor real
recovery_true = recovery_data['rougher.output.recovery']

# Calcular Error Absoluto Medio
mae = np.mean(np.abs(recovery_true - recovery_calc))
print(f"Error Absoluto Medio (EAM) en la recuperación calculada: {mae:.4f}")

# ----------------- 1.3 Analizar características no disponibles en el conjunto de prueba----------------
# Características que están en train pero no en test
missing_in_test = sorted(set(train) - set(test))

# Mostrar resultados
print("Columnas que están en el conjunto de entrenamiento pero no en el conjunto de prueba:")
for col in missing_in_test:
    print(col)

# Revisar sus tipos
print("\nTipos de datos de las columnas faltantes:")
print(train[missing_in_test].dtypes)

# ------------------- 1.2 Verificar cálculo de rougher.output.recovery -------------------
def calculate_recovery(C, F, T):
    return (C * (F - T)) / (F * (C - T)) * 100

# Calcular y comparar recuperación
cols_needed = ['rougher.output.concentrate_au', 'rougher.input.feed_au', 'rougher.output.tail_au', 'rougher.output.recovery']
recovery_data = train[cols_needed].dropna()
recovery_calc = calculate_recovery(
    recovery_data['rougher.output.concentrate_au'],
    recovery_data['rougher.input.feed_au'],
    recovery_data['rougher.output.tail_au']
)
recovery_true = recovery_data['rougher.output.recovery']
mae = np.mean(np.abs(recovery_true - recovery_calc))
print(f"\n✅ Error Absoluto Medio (EAM): {mae:.4f}")

# ------------------- 1.3 Características no disponibles en test -------------------
missing_in_test = sorted(set(train) - set(test))
print("\nColumnas faltantes en test:")
for col in missing_in_test:
    print(col)
print("\nTipos de esas columnas:")
print(train[missing_in_test].dtypes)

# ------------------- 1.4 Preprocesamiento de datos -------------------
# Columnas objetivo
target_columns = ['rougher.output.recovery', 'final.output.recovery']

# Columnas comunes
feature_columns = list(set(train.columns) & set(test.columns))

# Separar features y targets
features_train = train[feature_columns].copy()
target_train = train[target_columns].copy()
features_test = test[feature_columns].copy()

# Limpiar features_train y sincronizar con target_train
features_train_clean = features_train.dropna()
target_train_clean = target_train.loc[features_train_clean.index]
target_train_clean = target_train_clean.dropna()
features_train_clean = features_train_clean.loc[target_train_clean.index]

# Imputar faltantes en test
features_test_clean = features_test.copy()
features_test_clean = features_test_clean.interpolate(method='linear')

# Recuperar target_test desde full
target_test = full.loc[features_test_clean.index, target_columns]
target_test = target_test.dropna()

# Sincronizar índices
features_test_clean = features_test_clean.loc[target_test.index]

# Eliminar cualquier fila con NaN remanente
test_limpio = pd.concat([features_test_clean, target_test], axis=1).dropna()
features_test_clean = test_limpio[features_test_clean.columns]
target_test = test_limpio[target_test.columns]

# Verificación final
print("\n✅ Faltantes en features_train_clean:", features_train_clean.isna().sum().sum())
print("✅ Faltantes en target_train_clean:", target_train_clean.isna().sum().sum())
print("✅ Faltantes en features_test_clean:", features_test_clean.isna().sum().sum())
print("✅ Faltantes en target_test:", target_test.isna().sum().sum())
print("✅ Filas en features_train_clean:", len(features_train_clean))
print("✅ Filas en features_test_clean:", len(features_test_clean))
print("✅ Columnas iguales:", list(features_train_clean.columns) == list(features_test_clean.columns))


# ------------ 2.1. Observa cómo cambia la concentración de metales (Au, Ag, Pb) en función de la etapa de purificación. ----------------
def analizar_concentraciones_metales(train):
    metales = ['au', 'ag', 'pb']
    etapas = {
        'Rougher Feed': 'rougher.input.feed_',
        'Rougher': 'rougher.output.concentrate_',
        'Primary Cleaner': 'primary_cleaner.output.concentrate_',
        'Final': 'final.output.concentrate_'
    }

    # DISTRIBUCIÓN DE CONCENTRACIONES 
    for metal in metales:
        print(f"\n📌 CONCENTRACIÓN DE {metal.upper()}")
        plt.figure(figsize=(10, 6))

        for nombre_etapa, prefijo in etapas.items():
            columna = prefijo + metal
            if columna in train.columns:
                concentracion = train[columna].dropna()
                media = concentracion.mean()
                std = concentracion.std()

                print(f"✅ Etapa: {nombre_etapa}")
                print(f"  Media: {media:.2f}")
                print(f"  Desviación estándar: {std:.2f}")
                print(f"  Valores iniciales:\n{concentracion.head()}\n")

                sns.histplot(concentracion, 
                             label=nombre_etapa, 
                             kde=False, 
                             bins=50, 
                             alpha=0.5)

        plt.title(f'Distribución de concentración de {metal.upper()} en cada etapa')
        plt.xlabel('Concentración (%)')
        plt.ylabel('Frecuencia')
        plt.legend(title='Etapa')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # TENDENCIA DE LA CONCENTRACIÓN MEDIA
    datos = []
    for metal in metales:
        for nombre_etapa, prefijo in etapas.items():
            columna = prefijo + metal
            if columna in train.columns:
                media = train[columna].mean()
                datos.append({
                    'Metal': metal.upper(),
                    'Etapa': nombre_etapa,
                    'Concentración Media': media
                })

    df_tendencia = pd.DataFrame(datos)
    orden_etapas = ['Rougher Feed', 'Rougher', 'Primary Cleaner', 'Final']
    df_tendencia['Etapa'] = pd.Categorical(df_tendencia['Etapa'], categories=orden_etapas, ordered=True)
    df_tendencia = df_tendencia.sort_values(by=['Metal', 'Etapa'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_tendencia,
                 x='Etapa',
                 y='Concentración Media',
                 hue='Metal',
                 marker='o',
                 linewidth=2.5)

    for _, fila in df_tendencia.iterrows():
        plt.text(x=fila['Etapa'],
                 y=fila['Concentración Media'] + 0.5,
                 s=f"{fila['Concentración Media']:.2f}",
                 ha='center',
                 fontsize=9)

    plt.title('Tendencia de la concentración media de metales por etapa del proceso')
    plt.ylabel('Concentración Media (%)')
    plt.xlabel('Etapa del proceso')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Metal')
    plt.tight_layout()
    plt.show()

# Ejecutar todo
analizar_concentraciones_metales(train)

# ------------------ 2.2. Comparación de las distribuciones del tamaño de partículas de la alimentación en el conjunto de entrenamiento y prueba ----------------
def comparar_distribuciones_feed_size(train, test):
    columnas = ['rougher.input.feed_size', 'primary_cleaner.input.feed_size']
    
    for col in columnas:
        if col in train.columns and col in test.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(train[col], label='Entrenamiento', kde=True, stat='density', bins=50, color='blue', alpha=0.5)
            sns.histplot(test[col], label='Prueba', kde=True, stat='density', bins=50, color='orange', alpha=0.5)
            plt.title(f'Distribución del tamaño de partícula: {col}')
            plt.xlabel('Tamaño de partícula')
            plt.ylabel('Densidad')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Mostrar estadísticas
            print(f"\nColumna: {col}")
            print(f"Entrenamiento - Media: {train[col].mean():.2f}, Desviación estándar: {train[col].std():.2f}")
            print(f"Prueba - Media: {test[col].mean():.2f}, Desviación estándar: {test[col].std():.2f}")
        else:
            print(f"La columna '{col}' no está en ambos conjuntos.")

# Ejecutar la comparación
comparar_distribuciones_feed_size(train, test)


# ----------------- 2.3. Análisis de concentraciones totales de sustancias en las etapas -----------------
def analizar_concentraciones_totales(data, etapas, sustancias):
    for etapa in etapas:
        columnas = [f"{etapa}_{s}" for s in sustancias if f"{etapa}_{s}" in data.columns]
        if columnas:
            total = data[columnas].sum(axis=1)
            fuera_de_rango = total[(total > 100) | (total < 0)]

            print(f"\n 🔹 {etapa.upper()} 🔹")
            print(f"Número de muestras: {len(total)}")
            print(f"Muestras fuera del rango físico (0–100%): {len(fuera_de_rango)}")
            print(f"Máxima concentración total: {total.max():.2f}")
            print(f"Mínima concentración total: {total.min():.2f}")

            # Histograma
            plt.figure(figsize=(8, 5))
            sns.histplot(total, bins=100, kde=True)
            plt.axvline(100, color='red', linestyle='--')
            plt.axvline(0, color='red', linestyle='--')
            plt.title(f'Total de concentraciones en {etapa}')
            plt.xlabel('Concentración total (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
# Ejecutar
etapas = ['rougher.input.feed', 'rougher.output.concentrate', 'final.output.concentrate']
sustancias = ['au', 'ag', 'pb', 'sol']
analizar_concentraciones_totales(train, etapas, sustancias)

# Eliminar valores fuera del rango físico (concentración total < 0 o > 100):
def eliminar_anomalías_concentración(data, etapas, sustancias):
    indices_validos = data.index
    for etapa in etapas:
        columnas = [f"{etapa}_{s}" for s in sustancias if f"{etapa}_{s}" in data.columns]
        if columnas:
            total = data[columnas].sum(axis=1)
            indices_validos = indices_validos.intersection(total[(total >= 0) & (total <= 100)].index)
    return data.loc[indices_validos]

# Aplicar limpieza física a train y test
train_filtrado = eliminar_anomalías_concentración(train, etapas, sustancias)
test_filtrado = eliminar_anomalías_concentración(test, etapas, sustancias)

def filtrar_outliers_iqr(data, columnas):
    total = data[columnas].sum(axis=1)
    Q1 = total.quantile(0.25)
    Q3 = total.quantile(0.75)
    IQR = Q3 - Q1
    filtro = (total >= (Q1 - 1.5 * IQR)) & (total <= (Q3 + 1.5 * IQR))
    return data.loc[filtro]

# Aplicar IQR solo sobre concentraciones del concentrado final
cols_concentrado_final = [f"final.output.concentrate_{s}" for s in sustancias if f"final.output.concentrate_{s}" in train_filtrado.columns]
datos_filtrados = filtrar_outliers_iqr(train_filtrado, cols_concentrado_final)

# Resultado final de limpieza 
# Este será tu conjunto limpio
train = datos_filtrados.copy()
test = test_filtrado.copy()

print("✅ Tamaño final del train limpio:", train.shape)
print("✅ Tamaño final del test limpio:", test.shape)

# -------------- Construye el modelo --------------
# Función para calcular el valor final de sMAPE
# Funciones de métrica sMAPE
# Calcular sMAPE
def smape(y_true, y_pred):
    denominator = (abs(y_true) + abs(y_pred)) / 2
    diff = abs(y_true - y_pred) / denominator
    return 100 * diff.mean()

def final_smape(smape_r, smape_f):
    return 0.25 * smape_r + 0.75 * smape_f

# ------------ Entrenamiento de diferentes modelos y evaluación ------------
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff[denominator != 0]) * 100

def final_smape(smape_r, smape_f):
    return 0.25 * smape_r + 0.75 * smape_f

modelos = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Decision Tree (depth=10)": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest (depth=10)": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Random Forest (depth=10, trees=200)": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Random Forest (depth=15)": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
}

for nombre, modelo in modelos.items():
    try:
        modelo.fit(features_train_clean, target_train_clean)

        pred_rougher = modelo.predict(features_test_clean)[:, 0]
        pred_final = modelo.predict(features_test_clean)[:, 1]

        smape_r = smape(target_test['rougher.output.recovery'], pred_rougher)
        smape_f = smape(target_test['final.output.recovery'], pred_final)
        smape_total = final_smape(smape_r, smape_f)

        print(f"\n✅ Modelo: {nombre}")
        print(f"sMAPE Rougher: {smape_r:.2f}")
        print(f"sMAPE Final:   {smape_f:.2f}")
        print(f"sMAPE Total:   {smape_total:.2f}")

    except Exception as e:
        print(f"\n❌ Error con el modelo {nombre}: {e}")

