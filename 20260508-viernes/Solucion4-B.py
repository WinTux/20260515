# ==========================================
# 1. IMPORTACIONES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==========================================
# 2. CARGA Y LIMPIEZA TOTAL DE DATOS
# ==========================================
ruta = 'C:/Users/PAS/Desktop/DataSet/all_games.csv'
df_final = pd.read_csv(ruta)

# Limpieza de valores 'tbd' y conversión a numérico
df_final['user_review'] = pd.to_numeric(df_final['user_review'], errors='coerce')
df_final = df_final.dropna(subset=['meta_score', 'user_review'])

# Limpieza de nombres de plataforma
df_final['platform'] = df_final['platform'].str.strip()

# ==========================================
# 3. PREPARACIÓN Y ESCALADO
# ==========================================
# Seleccionamos las columnas para agrupar
features = df_final[['meta_score', 'user_review']]

# Es vital escalar para que K-Means funcione correctamente
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ==========================================
# 4. ENTRENAMIENTO DEL MODELO (K-Means)
# ==========================================
# Entrenamos con k=3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final['cluster'] = kmeans.fit_predict(features_scaled)

# ==========================================
# 5. MÉTODO PARA PROBAR EL MODELO (Asignar nuevos datos)
# ==========================================
def probar_segmentacion(meta_score_nuevo, user_review_nuevo):
    """
    Recibe puntuaciones de un juego nuevo y determina a qué cluster pertenece.
    """
    # 1. Crear el vector de entrada
    datos_nuevos = np.array([[meta_score_nuevo, user_review_nuevo]])
    
    # 2. ESCALAR los datos nuevos con el mismo transformador usado en el entrenamiento
    datos_escalados = scaler.transform(datos_nuevos)
    
    # 3. Predecir el cluster
    cluster_asignado = kmeans.predict(datos_escalados)[0]
    
    print(f"\n>>> Análisis de Segmentación:")
    print(f"    Entrada: MetaScore {meta_score_nuevo}, UserReview {user_review_nuevo}")
    print(f"    Segmento Asignado: Cluster {cluster_asignado}")
    
    # Explicación lógica del grupo (basado en centroides)
    if cluster_asignado == 0:
        print("    Perfil: Juego de recepción mixta o promedio.")
    elif cluster_asignado == 1:
        print("    Perfil: Obra Maestra / Blockbuster (Altas notas en ambos).")
    else:
        print("    Perfil: Posible Juego de Culto (Diferencia entre crítica y fans).")

# ==========================================
# 6. EVALUACIÓN Y VISUALIZACIÓN
# ==========================================
# Métrica de calidad técnica
score = silhouette_score(features_scaled, df_final['cluster'])
print(f"--- VALIDACIÓN TÉCNICA ---")
print(f"Coeficiente de Silueta: {score:.4f}")

# Pruebas manuales con datos inventados
probar_segmentacion(95, 9.0)  # Debería ser Cluster de Éxito
probar_segmentacion(60, 5.0)  # Debería ser Cluster Promedio

# Graficar resultados
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_final, x='meta_score', y='user_review', 
                hue='cluster', palette='viridis', alpha=0.5)
plt.title("Fase 4-B: Segmentación de Videojuegos (K-Means)")
plt.show()