# =========================================================
# NORMALIZACIÓN PASO A PASO CON GRÁFICAS
# =========================================================

# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# =========================================================
# CREACIÓN DEL DATASET
# =========================================================

datos = pd.DataFrame({

    'tenure': [12, 48, 5, 60, 25],

    'MonthlyCharges': [80, 120, 40, 95, 60],

    'TotalCharges': [1200, 5400, 300, 7000, 2500]

})

# =========================================================
# MOSTRAR DATOS ORIGINALES
# =========================================================

print("\n================================================")
print("DATOS ORIGINALES")
print("================================================\n")

print(datos)

# =========================================================
# CREAR EL SCALER
# =========================================================

scaler = StandardScaler()

# =========================================================
# APRENDER LOS DATOS
# =========================================================

scaler.fit(datos)

# =========================================================
# MOSTRAR MEDIA
# =========================================================

print("\n================================================")
print("MEDIA DE LAS VARIABLES")
print("================================================\n")

for columna, media in zip(datos.columns, scaler.mean_):

    print(f"{columna}: {media}")

# =========================================================
# MOSTRAR DESVIACIÓN ESTÁNDAR
# =========================================================

print("\n================================================")
print("DESVIACIÓN ESTÁNDAR")
print("================================================\n")

for columna, desviacion in zip(datos.columns, scaler.scale_):

    print(f"{columna}: {desviacion}")

# =========================================================
# NORMALIZAR LOS DATOS
# =========================================================

datos_normalizados = scaler.transform(datos)

# =========================================================
# CONVERTIR A DATAFRAME
# =========================================================

datos_normalizados_df = pd.DataFrame(
    datos_normalizados,
    columns=datos.columns
)

# =========================================================
# MOSTRAR DATOS NORMALIZADOS
# =========================================================

print("\n================================================")
print("DATOS NORMALIZADOS")
print("================================================\n")

print(datos_normalizados_df)

# =========================================================
# GRÁFICA 1 -> DATOS ORIGINALES
# =========================================================

plt.figure(figsize=(10,5))

for columna in datos.columns:

    plt.plot(datos[columna], marker='o', label=columna)

plt.title("DATOS ORIGINALES")

plt.xlabel("Registros")

plt.ylabel("Valores")

plt.legend()

plt.grid(True)

plt.show()

# =========================================================
# GRÁFICA 2 -> DATOS NORMALIZADOS
# =========================================================

plt.figure(figsize=(10,5))

for columna in datos_normalizados_df.columns:

    plt.plot(
        datos_normalizados_df[columna],
        marker='o',
        label=columna
    )

plt.title("DATOS NORMALIZADOS")

plt.xlabel("Registros")

plt.ylabel("Valores Escalados")

plt.legend()

plt.grid(True)

plt.show()

# =========================================================
# COMPARACIÓN VISUAL
# =========================================================

print("\n================================================")
print("INTERPRETACIÓN DE LAS GRÁFICAS")
print("================================================\n")

print("ANTES DE NORMALIZAR:")
print("- Las variables tenían escalas muy diferentes")
print("- TotalCharges dominaba los valores")

print("\nDESPUÉS DE NORMALIZAR:")
print("- Todas las variables quedaron en escalas similares")
print("- Los datos están centrados alrededor de 0")
print("- El modelo puede aprender mejor")

# =========================================================
# MENSAJE FINAL
# =========================================================

print("\n================================================")
print("PROCESO DE NORMALIZACIÓN Y GRÁFICAS FINALIZADO")
print("================================================\n")