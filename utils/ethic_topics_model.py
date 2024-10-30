import spacy
import numpy as np
import json
from utils.bertopic_model import cargar_stopwords, StemmerTokenizer, stop_words_custom

# Cargar el modelo de embeddings de spaCy en español
nlp = spacy.load("es_core_news_md") 

# Lematizar el diccionario
def lematizar_keywords_dict():
    path = 'dictionaries/ethic_topics.json'
    with open(path, 'r', encoding='utf-8') as f:
        keywords_dict = json.load(f)
    
    lematizado_dict = {}
    tokenizer = StemmerTokenizer(lemmatize=True)
    
    for key, values in keywords_dict.items():
        lematizado_key = key  
        lematizado_values = [tokenizer(value)[0] for value in values]  
        lematizado_dict[lematizado_key] = lematizado_values
    
    return lematizado_dict

# Lematizar el diccionario de palabras clave
keywords_dict = lematizar_keywords_dict()

def ethic_palabras_clave_en_comentario(comentario):
    ponderaciones = {}
    doc = nlp(comentario.lower())
    
    for token in doc:
        if token.is_alpha and token.text not in stop_words_custom:
            for key, values in keywords_dict.items():
                # Verifica coincidencia directa de la lematización
                if token.lemma_ in values:
                    ponderaciones[key] = ponderaciones.get(key, 0) + 1

    # Si no hay coincidencias, agrega "Sin tópico"
    if not ponderaciones:
        ponderaciones["Sin tópico"] = 1

    return ponderaciones



def predecir_topicos_eticos(comentario):
    ponderaciones = ethic_palabras_clave_en_comentario(comentario)
    return [topic for topic, peso in ponderaciones.items() if peso > 0]  

def predict_ethic_topic(df1, df2, caso):
    df1['ETHIC_topicos_ind1'] = ""
    df1['ETHIC_topicos_ind2'] = ""

    df2['ETHIC_topicos_ind1'] = ""
    df2['ETHIC_topicos_ind2'] = ""

    if 'ETHIC_topicos_grup' in df1.columns:
        df1 = df1.drop(columns=['ETHIC_topicos_grup'])
    if 'ETHIC_topicos_grup' in df2.columns:
        df2 = df2.drop(columns=['ETHIC_topicos_grup'])

    # Definir una función para predecir tópicos de grupos
    def predecir_topicos_grup(df, group_cols, comment_col):
        grouped = df[group_cols + [comment_col]].drop_duplicates(subset=group_cols)
        grouped['ETHIC_topicos_grup'] = grouped[comment_col].apply(lambda x: predecir_topicos_eticos(str(x)))
        df = df.merge(grouped[group_cols + ['ETHIC_topicos_grup']], on=group_cols, how='left')
        return df

    # Topicos grupales
    df1 = predecir_topicos_grup(df1, ['Grup', 'seccion', 'agno'], 'Comentario - Grup - Diferencial 1')
    df2 = predecir_topicos_grup(df2, ['Grup', 'seccion', 'agno'], 'Comentario - Grup - Diferencial 2')

    # Topicos para Ind1 e Ind2
    for i, comentario in df1['Comentario - Ind1 - Diferencial 1'].items():
        topicos = predecir_topicos_eticos(comentario)
        df1.at[i, 'ETHIC_topicos_ind1'] = topicos

    for i, comentario in df1['Comentario - Ind2 - Diferencial 1'].items():
        topicos = predecir_topicos_eticos(comentario)
        df1.at[i, 'ETHIC_topicos_ind2'] = topicos

    for i, comentario in df2['Comentario - Ind1 - Diferencial 2'].items():
        topicos = predecir_topicos_eticos(comentario)
        df2.at[i, 'ETHIC_topicos_ind1'] = topicos

    for i, comentario in df2['Comentario - Ind2 - Diferencial 2'].items():
        topicos = predecir_topicos_eticos(comentario)
        df2.at[i, 'ETHIC_topicos_ind2'] = topicos

    # Guardar topicos predichos
    df1.to_csv(f"processed_data/{caso}/ETHIC_Topics_df1.csv", index=False)
    df2.to_csv(f"processed_data/{caso}/ETHIC_Topics_df2.csv", index=False)