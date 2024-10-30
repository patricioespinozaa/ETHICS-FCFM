import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import os

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