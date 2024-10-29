import re
import spacy
import pandas as pd
from nltk.stem import SnowballStemmer
from bertopic import BERTopic

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