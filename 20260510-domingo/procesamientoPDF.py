# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================

import os
import re
import unicodedata

import nltk
import chardet
import pdfplumber

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from langdetect import detect

from wordcloud import WordCloud

from nltk.corpus import stopwords

from nltk.tokenize import (
    word_tokenize,
    sent_tokenize
)

from nltk.stem import (
    SnowballStemmer,
    WordNetLemmatizer
)

from sklearn.feature_extraction.text import (
    CountVectorizer
)


# =========================================================
# DESCARGA DE RECURSOS NLTK
# =========================================================

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

ARCHIVO_PDF = (
    "C:/Users/PAS/Desktop/Min_texto/MineriaTexto.pdf"
)

MAX_CARACTERES = 3000


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def mostrar_titulo(texto):

    """
    Mostrar títulos decorativos.
    """

    print("\n")
    print("┌" + "─" * 70 + "┐")
    print(f"│ {texto:<68} │")
    print("└" + "─" * 70 + "┘")


# =========================================================

def normalizar_texto(texto):

    """
    Normalización textual:
    - minúsculas
    - eliminación de acentos
    - espacios redundantes
    """

    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(

        c

        for c in texto

        if not unicodedata.combining(c)

    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


# =========================================================

def generar_nube_palabras(texto):

    """
    Generación de nube de palabras.
    """

    wordcloud = WordCloud(

        width=1400,
        height=700,
        background_color="white",
        colormap="viridis",
        max_words=100

    ).generate(texto)

    plt.figure(figsize=(16, 8))

    plt.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    plt.axis("off")

    plt.title(
        "NUBE DE PALABRAS - MINERÍA DE TEXTOS",
        fontsize=20
    )

    plt.show()


# =========================================================
# ETAPA 0 - PREPARACIÓN DE FUENTES
# =========================================================

mostrar_titulo(
    "ETAPA 0 - PREPARACIÓN DE FUENTES"
)


# =========================================================
# VERIFICACIÓN DE EXISTENCIA DEL PDF
# =========================================================

if not os.path.exists(ARCHIVO_PDF):

    raise FileNotFoundError(
        f"No se encontró el archivo: {ARCHIVO_PDF}"
    )


# =========================================================
# CONTROL DE TAMAÑO DEL ARCHIVO
# =========================================================

tamaño_bytes = os.path.getsize(
    ARCHIVO_PDF
)

tamaño_mb = tamaño_bytes / (1024 * 1024)


# =========================================================
# ETAPA 1 - ADQUISICIÓN DE FUENTES
# =========================================================

mostrar_titulo(
    "ETAPA 1 - ADQUISICIÓN DE FUENTES"
)


# =========================================================
# EXTRACCIÓN DE TEXTO DESDE PDF
# =========================================================

texto_original = ""

with pdfplumber.open(ARCHIVO_PDF) as pdf:

    for pagina in pdf.pages:

        contenido = pagina.extract_text()

        if contenido:

            texto_original += contenido + " "


# =========================================================
# VERIFICACIÓN DE CODIFICACIÓN
# =========================================================

codificacion = chardet.detect(
    texto_original.encode()
)["encoding"]


# =========================================================
# DETECCIÓN DE IDIOMA
# =========================================================

idioma_detectado = detect(
    texto_original
)


# =========================================================
# ETAPA 2 - LIMPIEZA DEL TEXTO
# =========================================================

mostrar_titulo(
    "ETAPA 2 - LIMPIEZA DEL TEXTO"
)


# =========================================================
# LIMPIEZA BÁSICA
# =========================================================

texto_limpio = re.sub(
    r"\n",
    " ",
    texto_original
)

texto_limpio = re.sub(
    r"[^a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ\s]",
    " ",
    texto_limpio
)

texto_limpio = re.sub(
    r"\s+",
    " ",
    texto_limpio
).strip()


# =========================================================
# ELIMINACIÓN DE RUIDO
# =========================================================

texto_limpio = re.sub(
    r"https?:\/\/\S+",
    "",
    texto_limpio
)

texto_limpio = re.sub(
    r"\d+",
    " ",
    texto_limpio
)

texto_limpio = re.sub(
    r"\s+",
    " ",
    texto_limpio
).strip()


# =========================================================
# NORMALIZACIÓN
# =========================================================

texto_normalizado = normalizar_texto(
    texto_limpio
)


# =========================================================
# TOKENIZACIÓN
# =========================================================

tokens_palabras = word_tokenize(
    texto_normalizado
)

tokens_oraciones = sent_tokenize(
    texto_normalizado
)


# =========================================================
# ELIMINACIÓN DE STOPWORDS
# =========================================================

stopwords_es = set(
    stopwords.words("spanish")
)

tokens_sin_stopwords = [

    palabra

    for palabra in tokens_palabras

    if palabra not in stopwords_es

]


# =========================================================
# DOCUMENTO FINAL PROCESADO
# =========================================================

documento_final = " ".join(
    tokens_sin_stopwords
)


# =========================================================
# ETAPA 3 - PREPARACIÓN PARA EL ANÁLISIS
# =========================================================

mostrar_titulo(
    "ETAPA 3 - PREPARACIÓN PARA EL ANÁLISIS"
)


# =========================================================
# STEMMING
# =========================================================

stemmer = SnowballStemmer(
    "spanish"
)

stems = [

    stemmer.stem(palabra)

    for palabra in tokens_sin_stopwords

]


# =========================================================
# LEMATIZACIÓN
# =========================================================

lemmatizer = WordNetLemmatizer()

lemmas = [

    lemmatizer.lemmatize(palabra)

    for palabra in tokens_sin_stopwords

]


# =========================================================
# MATRIZ DTM
# =========================================================

corpus = [documento_final]

vectorizer = CountVectorizer()

dtm = vectorizer.fit_transform(
    corpus
)

vocabulario = vectorizer.get_feature_names_out()


# =========================================================
# MATRIZ TDM
# =========================================================

tdm = dtm.T


# =========================================================
# FRECUENCIA DE TÉRMINOS
# =========================================================

frecuencia_terminos = dtm.toarray().sum(
    axis=0
)


# =========================================================
# DATAFRAMES
# =========================================================

df_dtm = pd.DataFrame(

    dtm.toarray(),
    columns=vocabulario

)

df_tdm = pd.DataFrame(

    tdm.toarray(),
    index=vocabulario,
    columns=["Documento_1"]

)

df_frecuencias = pd.DataFrame({

    "Término": vocabulario,
    "Frecuencia": frecuencia_terminos

})

df_frecuencias = df_frecuencias.sort_values(

    by="Frecuencia",
    ascending=False

)


# =========================================================
# RESULTADOS FINALES
# =========================================================

print("\n")
print("=" * 100)
print("RESULTADOS FINALES DEL PROCESAMIENTO DE MINERÍA DE TEXTOS")
print("=" * 100)


# =========================================================
# RESULTADO 1 - CONTROL DE TAMAÑO
# =========================================================

mostrar_titulo(
    "[1] CONTROL DE TAMAÑO DEL DOCUMENTO"
)

print(f"\nTamaño del archivo : {tamaño_mb:.2f} MB")
print(f"Tamaño en bytes    : {tamaño_bytes} bytes")


# =========================================================
# RESULTADO 2 - CODIFICACIÓN
# =========================================================

mostrar_titulo(
    "[2] VERIFICACIÓN DE CODIFICACIÓN"
)

print(f"\nCodificación detectada : {codificacion}")


# =========================================================
# RESULTADO 3 - IDIOMA
# =========================================================

mostrar_titulo(
    "[3] DETECCIÓN DE IDIOMA"
)

print(f"\nIdioma detectado : {idioma_detectado}")


# =========================================================
# RESULTADO 4 - TEXTO ORIGINAL
# =========================================================

mostrar_titulo(
    "[4] TEXTO ORIGINAL EXTRAÍDO"
)

print(texto_original[:MAX_CARACTERES])


# =========================================================
# RESULTADO 5 - TEXTO LIMPIO
# =========================================================

mostrar_titulo(
    "[5] TEXTO LIMPIO"
)

print(texto_limpio[:MAX_CARACTERES])


# =========================================================
# RESULTADO 6 - TEXTO NORMALIZADO
# =========================================================

mostrar_titulo(
    "[6] TEXTO NORMALIZADO"
)

print(texto_normalizado[:MAX_CARACTERES])


# =========================================================
# RESULTADO 7 - TOKENIZACIÓN
# =========================================================

mostrar_titulo(
    "[7] TOKENIZACIÓN"
)

print(f"\nCantidad total de tokens: {len(tokens_palabras)}")

print("\nPrimeros 100 tokens:\n")

print(tokens_palabras[:100])


# =========================================================
# RESULTADO 8 - STOPWORDS
# =========================================================

mostrar_titulo(
    "[8] TOKENS SIN STOPWORDS"
)

print(
    f"\nCantidad de tokens limpios: "
    f"{len(tokens_sin_stopwords)}"
)

print("\nPrimeros 100 tokens:\n")

print(tokens_sin_stopwords[:100])


# =========================================================
# RESULTADO 9 - DOCUMENTO FINAL
# =========================================================

mostrar_titulo(
    "[9] DOCUMENTO FINAL PROCESADO"
)

print(documento_final[:5000])


# =========================================================
# RESULTADO 10 - STEMMING
# =========================================================

mostrar_titulo(
    "[10] RESULTADOS DE STEMMING"
)

print(
    f"\n{'PALABRA ORIGINAL':<35}"
    f"{'STEM':<20}"
)

print("-" * 60)

for original, stem in zip(
    tokens_sin_stopwords[:30],
    stems[:30]
):

    print(f"{original:<35}{stem:<20}")


# =========================================================
# RESULTADO 11 - LEMATIZACIÓN
# =========================================================

mostrar_titulo(
    "[11] RESULTADOS DE LEMATIZACIÓN"
)

print(
    f"\n{'PALABRA ORIGINAL':<35}"
    f"{'LEMA':<20}"
)

print("-" * 60)

for original, lema in zip(
    tokens_sin_stopwords[:30],
    lemmas[:30]
):

    print(f"{original:<35}{lema:<20}")


# =========================================================
# RESULTADO 12 - VOCABULARIO
# =========================================================

mostrar_titulo(
    "[12] VOCABULARIO EXTRAÍDO"
)

print(
    f"\nCantidad de términos únicos: "
    f"{len(vocabulario)}"
)

print("\nPrimeros 100 términos:\n")

for i, termino in enumerate(
    vocabulario[:100],
    start=1
):

    print(f"{i:>3}. {termino}")


# =========================================================
# RESULTADO 13 - MATRIZ DTM
# =========================================================

mostrar_titulo(
    "[13] MATRIZ DOCUMENTO-TÉRMINO (DTM)"
)

print("\n")

print(df_dtm)


# =========================================================
# RESULTADO 14 - MATRIZ TDM
# =========================================================

mostrar_titulo(
    "[14] MATRIZ TÉRMINO-DOCUMENTO (TDM)"
)

print("\n")

print(df_tdm)


# =========================================================
# RESULTADO 15 - FRECUENCIA DE TÉRMINOS
# =========================================================

mostrar_titulo(
    "[15] FRECUENCIA DE TÉRMINOS"
)

print("\n")

print(df_frecuencias.head(50))


# =========================================================
# RESULTADO 16 - NUBE DE PALABRAS
# =========================================================

mostrar_titulo(
    "[16] NUBE DE PALABRAS"
)

generar_nube_palabras(
    documento_final
)


# =========================================================
# RESUMEN FINAL
# =========================================================

print("\n")
print("=" * 100)
print("RESUMEN DEL PROCESAMIENTO")
print("=" * 100)

print("""

ETAPAS EJECUTADAS:

✓ ETAPA 0 - Preparación de fuentes
✓ ETAPA 1 - Adquisición de fuentes
✓ ETAPA 2 - Limpieza del texto
✓ ETAPA 3 - Preparación para el análisis

PROCESOS REALIZADOS:

✓ Control de tamaño
✓ Extracción de texto PDF
✓ Verificación de codificación
✓ Detección de idioma
✓ Limpieza básica
✓ Eliminación de ruido
✓ Normalización textual
✓ Tokenización
✓ Eliminación de stopwords
✓ Implementación de stemming
✓ Implementación de lematización
✓ Construcción de matriz DTM
✓ Construcción de matriz TDM
✓ Frecuencia de términos
✓ Nube de palabras

RESULTADO FINAL:

El documento PDF fue transformado correctamente
en un corpus textual estructurado y preparado para:

• Minería de textos
• NLP
• Clasificación documental
• Clustering
• Análisis semántico
• Ciencia de datos
• Aprendizaje automático

""")

print("=" * 100)
print("PROCESAMIENTO FINALIZADO CORRECTAMENTE")
print("=" * 100)