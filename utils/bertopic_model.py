import re
import spacy
import pandas as pd
from nltk.stem import SnowballStemmer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

nlp = spacy.load('es_core_news_md')

# Función para cargar las stopwords
def cargar_stopwords(ruta_archivo):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        print(f"El archivo {ruta_archivo} no se encontró. Usando un conjunto vacío.")
        return set()
    
global stop_words_custom
stop_words_custom = cargar_stopwords('dictionaries/stopwords_es.txt')

# Clase para tokenización y stemming/lemmatización
class StemmerTokenizer:
    def __init__(self, stem=False, lemmatize=True):
        self.stem = stem
        self.lemmatize = lemmatize
        self.ps = SnowballStemmer('spanish') if stem else None

    def __call__(self, doc):
        doc = re.sub(r'[^A-Za-záéíóúñÁÉÍÓÚÑ\s]', '', doc).lower()
        spacy_doc = nlp(doc)
        tokens = [
            self.ps.stem(token.lemma_) if self.stem and self.ps else token.lemma_
            for token in spacy_doc 
            if token.text not in stop_words_custom and not token.is_punct
        ]
        return tokens

# Función para cargar y preprocesar comentarios
def cargar_y_preprocesar_comentarios(df_dif1, df_dif2, tokenizer):
    columnas_comentarios_dif1 = [
        'Comentario - Ind1 - Diferencial 1',
        'Comentario - Grup - Diferencial 1',
        'Comentario - Ind2 - Diferencial 1'
    ]
    columnas_comentarios_dif2 = [
        'Comentario - Ind1 - Diferencial 2',
        'Comentario - Grup - Diferencial 2',
        'Comentario - Ind2 - Diferencial 2'
    ]
    
    comentarios_dif1 = df_dif1[columnas_comentarios_dif1].fillna('').values.flatten()
    comentarios_dif2 = df_dif2[columnas_comentarios_dif2].fillna('').values.flatten()
    comentarios = pd.concat([pd.Series(comentarios_dif1), pd.Series(comentarios_dif2)], axis=0).values.flatten()
    comentarios = [str(c) for c in comentarios]
    return [' '.join(tokenizer(c)) for c in comentarios]

# Función para entrenar y mostrar resultados del modelo
def entrenar_modelo_bertopic(comentarios_tokenizados, min_topic_size=10, nr_topics=10):
    model = BERTopic(language="spanish", min_topic_size=min_topic_size, nr_topics=nr_topics)
    topics, probs = model.fit_transform(comentarios_tokenizados)
    
    print("Tópicos más importantes:")
    for i, topic in enumerate(model.get_topic_info().head(10)['Name']):
        print(f"Tópico {i}: {topic}")
    print("Largo de los tópicos:", len(model.get_topic_info()))
    
    model.save("models/BertTopic_model")
    return model

# === Obtencion de topicos === #

import pandas as pd
from collections import Counter

def predict_topics(model, df_dif1, df_dif2, caso):
    # Extraer tópicos para cada etapa (Ind1, Grup, Ind2)
    topics_ind1_dif1 = model.transform(df_dif1['Comentario - Ind1 - Diferencial 1'].values.flatten())[0]
    topics_grup_dif1 = model.transform(df_dif1['Comentario - Grup - Diferencial 1'].values.flatten())[0]
    topics_ind2_dif1 = model.transform(df_dif1['Comentario - Ind2 - Diferencial 1'].values.flatten())[0]

    topics_ind1_dif2 = model.transform(df_dif2['Comentario - Ind1 - Diferencial 2'].values.flatten())[0]
    topics_grup_dif2 = model.transform(df_dif2['Comentario - Grup - Diferencial 2'].values.flatten())[0]
    topics_ind2_dif2 = model.transform(df_dif2['Comentario - Ind2 - Diferencial 2'].values.flatten())[0]

    # Guardar los tópicos en los dataframes
    df_dif1['BERT_topicos_ind1'] = topics_ind1_dif1
    df_dif1['BERT_topicos_grup'] = topics_grup_dif1
    df_dif1['BERT_topicos_ind2'] = topics_ind2_dif1

    df_dif2['BERT_topicos_ind1'] = topics_ind1_dif2
    df_dif2['BERT_topicos_grup'] = topics_grup_dif2
    df_dif2['BERT_topicos_ind2'] = topics_ind2_dif2

    # Guardar como csv
    df_dif1.to_csv(f"processed_data/{caso}/BERT_df1.csv", index=False)
    df_dif2.to_csv(f"processed_data/{caso}/BERT_df2.csv", index=False)


def contar_topicos(caso):
    # Leer los csv
    df_dif1 = pd.read_csv(f"processed_data/{caso}/BERT_df1.csv")
    df_dif2 = pd.read_csv(f"processed_data/{caso}/BERT_df2.csv")

    # Contar la frecuencia de tópicos en cada grupo
    topic_counts = {
        'Ind1_Dif1': Counter(df_dif1['BERT_topicos_ind1']),
        'Grup_Dif1': Counter(df_dif1['BERT_topicos_grup']),
        'Ind2_Dif1': Counter(df_dif1['BERT_topicos_ind2']),
        'Ind1_Dif2': Counter(df_dif2['BERT_topicos_ind1']),
        'Grup_Dif2': Counter(df_dif2['BERT_topicos_grup']),
        'Ind2_Dif2': Counter(df_dif2['BERT_topicos_ind2'])
    }
    
    return topic_counts

# Función para generar un gráfico de barras agrupadas con Ind1, Grup, e Ind2
def graficar_topicos_agrupados(topic_counts, differential, model, caso):
    path = f"resultados/{caso}"
    
    # Construir las claves de conteo de tópicos basadas en el diferencial
    key_ind1 = f'Ind1_Dif{differential}'
    key_grup = f'Grup_Dif{differential}'
    key_ind2 = f'Ind2_Dif{differential}'

    # Extraer las frecuencias para las etapas, usando Counter si no existe
    topic_counts_ind1 = topic_counts.get(key_ind1, Counter())
    topic_counts_grup = topic_counts.get(key_grup, Counter())
    topic_counts_ind2 = topic_counts.get(key_ind2, Counter())

    # Obtener lista combinada de tópicos únicos en las tres fases
    all_topics = set(topic_counts_ind1.keys()).union(set(topic_counts_grup.keys()), set(topic_counts_ind2.keys()))
    
    # Filtrar tópicos que tienen una frecuencia total mayor a 10
    filtered_topics = [
        topic for topic in all_topics 
        if (topic_counts_ind1.get(topic, 0) + topic_counts_grup.get(topic, 0) + topic_counts_ind2.get(topic, 0)) > 10
    ]

    # Obtener palabras clave de los tópicos
    top_words = []
    for topic in filtered_topics:
        try:
            top_words.append(", ".join([w[0] for w in model.get_topic(topic)[:5]]))
        except:
            top_words.append(f"Tópico {topic} no encontrado")

    # Frecuencias de cada tópico en Ind1, Grup e Ind2
    freqs_ind1 = [topic_counts_ind1.get(topic, 0) for topic in filtered_topics]
    freqs_grup = [topic_counts_grup.get(topic, 0) for topic in filtered_topics]
    freqs_ind2 = [topic_counts_ind2.get(topic, 0) for topic in filtered_topics]

    # Posiciones en el eje X
    x = range(len(filtered_topics))

    # Crear el gráfico de barras agrupadas
    plt.figure(figsize=(14, 8))
    
    width = 0.2  # Ancho de las barras
    plt.bar([p - width for p in x], freqs_ind1, width=width, label='Ind1', color=sns.color_palette("Blues")[2])
    plt.bar(x, freqs_grup, width=width, label='Grup', color=sns.color_palette("Greens")[2])
    plt.bar([p + width for p in x], freqs_ind2, width=width, label='Ind2', color=sns.color_palette("Oranges")[2])

    # Añadir etiquetas y leyenda
    plt.xlabel('Tópicos (Palabras Clave)', fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.title(f"Frecuencia de Tópicos por Etapa - Diferencial {differential}", fontsize=16)
    plt.xticks(ticks=x, labels=top_words, rotation=90, fontsize=12)
    plt.legend(fontsize=12)

    # Añadir líneas de cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    # Guardar el gráfico
    plt.savefig(f"{path}/BERT_frec_topicos_d{differential}.png", dpi=300, bbox_inches='tight')

def BERT_contar_topicos_distintos(df, columna):
    # Crear un diccionario para almacenar los resultados
    topicos_distintos = 0
    topicos_distintos += df[columna].nunique()
    
    return topicos_distintos
