# ==========================================
# 1. IMPORTACIONES
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split

# ==========================================
# 2. CARGA Y LIMPIEZA TOTAL DE DATOS
# ==========================================
# Asegúrate de que el archivo 'all_games.csv' esté en esta carpeta
ruta = 'C:/Users/PAS/Desktop/DataSet/all_games.csv'
df_final = pd.read_csv(ruta)

# ELIMINACIÓN DE ESPACIOS: Esta línea limpia los nombres de plataformas de raíz
df_final['platform'] = df_final['platform'].str.strip()

# Crear etiqueta binaria: 1 para Éxito (Meta Score > 80), 0 para el resto
df_final['is_hit'] = (df_final['meta_score'] > 80).astype(int)

# Preparar variables independientes (X) y dependiente (y)
X = pd.get_dummies(df_final[['platform']], drop_first=True)
y = df_final['is_hit']

# División del dataset: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. ENTRENAMIENTO DEL MODELO
# ==========================================
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 4. MÉTODO PARA PROBAR EL MODELO
# ==========================================
def probar_modelo(nombre_consola):
    # Limpiamos la entrada del usuario
    nombre_consola = nombre_consola.strip()
    columna_buscada = f'platform_{nombre_consola}'
    
    # Creamos un DataFrame de entrada con la misma estructura que X
    entrada_usuario = pd.DataFrame(0, index=[0], columns=X.columns)
    
    if columna_buscada in entrada_usuario.columns:
        entrada_usuario[columna_buscada] = 1
        prediccion = clf.predict(entrada_usuario)[0]
        probabilidades = clf.predict_proba(entrada_usuario)[0]
        
        resultado = "ÉXITO" if prediccion == 1 else "PROMEDIO/BAJO"
        print(f"\n>>> Análisis para la consola: {nombre_consola}")
        print(f"    Resultado esperado: {resultado}")
        print(f"    Probabilidad de Éxito: {probabilidades[1]:.2%}")
    else:
        print(f"\n[!] Error: La consola '{nombre_consola}' no fue encontrada.")
        # Mostramos sugerencias basadas en los nombres ya limpios
        sugerencias = [c.replace('platform_', '') for c in X.columns[:5]]
        print(f"    Intenta con uno de estos nombres: {sugerencias}")

# ==========================================
# 5. EVALUACIÓN Y PRUEBAS
# ==========================================
# Reporte de métricas
y_pred = clf.predict(X_test)
print("--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred))

# Pruebas manuales (Ahora sin errores por espacios)
probar_modelo('	PlayStation 3')
probar_modelo('	Xbox 360')
probar_modelo('Swtich')

# Visualización del Árbol de Decisión
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['Bajo', 'Éxito'])
plt.title("Árbol de Decisión para el Éxito de Videojuegos")
plt.show()