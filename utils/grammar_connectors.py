import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from utils.bertopic_model import cargar_stopwords

# Cargar el modelo de Spacy en español
nlp = spacy.load("es_core_news_md")

def identificar_conectores(texto):
    """Identifica conectores en el texto dado."""
    doc = nlp(texto)
    conectores_en_texto = [token.text for token in doc if token.pos_ in ["CCONJ", "SCONJ", "ADP"]]
    return conectores_en_texto

def extract_grammar_connectors(df1, df2, caso):
    # Leer el archivo de datos
    conectores_totales = []

    # Extraer conectores de cada columna
    Ind1_d1 = df1['Comentario - Ind1 - Diferencial 1'].apply(identificar_conectores).tolist()
    Grup_d1 = df1['Comentario - Grup - Diferencial 1'].apply(identificar_conectores).tolist()
    Ind2_d1 = df1['Comentario - Ind2 - Diferencial 1'].apply(identificar_conectores).tolist()

    Ind1_d2 = df2['Comentario - Ind1 - Diferencial 2'].apply(identificar_conectores).tolist()
    Grup_d2 = df2['Comentario - Grup - Diferencial 2'].apply(identificar_conectores).tolist()
    Ind2_d2 = df2['Comentario - Ind2 - Diferencial 2'].apply(identificar_conectores).tolist()

    conectores_totales.extend([conector for sublist in Ind1_d1 for conector in sublist])
    conectores_totales.extend([conector for sublist in Grup_d1 for conector in sublist])
    conectores_totales.extend([conector for sublist in Ind2_d1 for conector in sublist])

    conectores_totales.extend([conector for sublist in Ind1_d2 for conector in sublist])
    conectores_totales.extend([conector for sublist in Grup_d2 for conector in sublist])
    conectores_totales.extend([conector for sublist in Ind2_d2 for conector in sublist])

    # Calcular la frecuencia de conectores
    frecuencia_conectores = Counter(conectores_totales)
    df_frecuencia = pd.DataFrame(frecuencia_conectores.items(), columns=['Conectores', 'Frecuencia'])
    df_frecuencia = df_frecuencia.sort_values(by='Frecuencia', ascending=False)

    # Guardar conectores y sus frecuencias en un JSON
    df_frecuencia.to_json('dictionaries/conectores_frecuencia.json', orient='records')

    # Crear el directorio si no existe
    os.makedirs(f"resultados/{caso}", exist_ok=True)

    # Crear una tabla con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # Ocultar los ejes
    table = ax.table(cellText=df_frecuencia.values, colLabels=df_frecuencia.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Guardar la tabla como imagen
    plt.savefig(f"resultados/{caso}/Conectores_frecuencia_tabla.png")


# === Analisis de palabras clave después de conectores ===

import os
import pandas as pd
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nlp = spacy.load("es_core_news_md")

def identificar_conectores_y_frases(texto):
    doc = nlp(texto)
    conectores_y_frases = []
    for token in doc:
        if token.pos_ in ["CCONJ", "SCONJ", "ADP"]:
            frase = ' '.join([t.text for t in doc[token.i + 1:token.i + 6]])  # Captura hasta 5 palabras después del conector
            conectores_y_frases.append((token.text, frase))
    return conectores_y_frases

def extract_keywords_after_connectors(df1, df2, caso):
    conectores_y_frases_totales = []

    # Extraer conectores y frases de las columnas de df1 para Diferencial 1
    for column in ['Comentario - Ind1 - Diferencial 1', 'Comentario - Grup - Diferencial 1', 'Comentario - Ind2 - Diferencial 1']:
        conectores_y_frases_totales.extend(df1[column].apply(identificar_conectores_y_frases).tolist())

    # Extraer conectores y frases de las columnas de df2 para Diferencial 2
    for column in ['Comentario - Ind1 - Diferencial 2', 'Comentario - Grup - Diferencial 2', 'Comentario - Ind2 - Diferencial 2']:
        conectores_y_frases_totales.extend(df2[column].apply(identificar_conectores_y_frases).tolist())

    # Aplanar la lista de listas
    conectores_y_frases_totales = [item for sublist in conectores_y_frases_totales for item in sublist]

    # Contar frecuencia de conectores
    conectores = [item[0] for item in conectores_y_frases_totales]
    frecuencia_conectores = Counter(conectores)
    conectores_mas_usados = [item[0] for item in frecuencia_conectores.most_common(10)]
    frases_relevantes = [frase for conector, frase in conectores_y_frases_totales if conector in conectores_mas_usados]

    # Calcular TF-IDF para identificar palabras clave en las frases relevantes
    custom_stopwords = list(cargar_stopwords('dictionaries/stopwords_es.txt'))
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=20)  
    tfidf_matrix = vectorizer.fit_transform(frases_relevantes)

    # Obtener palabras clave
    palabras_clave = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keywords = sorted(zip(palabras_clave, tfidf_scores), key=lambda x: x[1], reverse=True)

    # Convertir palabras clave en DataFrame
    df_keywords = pd.DataFrame(keywords, columns=['Palabra Clave', 'Importancia'])
    
    # Guardar el DataFrame como CSV
    os.makedirs(f"resultados/{caso}", exist_ok=True)
    df_keywords.to_csv(f"processed_data/{caso}/Palabras_Clave_Despues_Conectores.csv", index=False)

    # Crear un gráfico de tabla para las palabras y su importancia
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    tabla = plt.table(cellText=df_keywords.values,
                      colLabels=df_keywords.columns,
                      cellLoc='center',
                      loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.2)
    
    plt.title("Palabras post conectores", fontsize=14)
    plt.savefig(f"resultados/{caso}/Conectores_Palabras_Clave.png", bbox_inches='tight')
    plt.close()

    print(f"Palabras clave después de conectores guardadas en resultados/{caso}/Palabras_Clave_Despues_Conectores.csv")
    return df_keywords


"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Cargar el tokenizador y el modelo
model_name = "PlanTL-GOB-ES/roberta-large-bne-capitel-pos"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
# Texto de ejemplo
texto = "A pesar de que el clima era adverso y las condiciones no eran las ideales, decidimos continuar con nuestra excursión, ya que, además de ser una experiencia única, nos permitiría disfrutar de la belleza del paisaje, mientras que, al mismo tiempo, fortalecería nuestros lazos como grupo"

# Tokenizar el texto
tokens = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)

# Realizar la inferencia
with torch.no_grad():
    outputs = model(**tokens)

# Obtener las etiquetas
logits = outputs.logits
predicciones = torch.argmax(logits, dim=2)

# Decodificar las etiquetas
etiquetas = [model.config.id2label[pred.item()] for pred in predicciones[0]]

# Mostrar los tokens y sus etiquetas
for token, etiqueta in zip(tokenizer.tokenize(texto), etiquetas):
    print(f"{token}: {etiqueta}")

# Filtrar conectores (si 'CONJ' es la etiqueta para conectores)
conectores = [token[1:] for token, etiqueta in zip(tokenizer.tokenize(texto), etiquetas) if etiqueta in ['CCONJ', 'ADP']]
conectores_unicos = list(set(conectores))

print("Conectores utilizados:", conectores_unicos)
"""