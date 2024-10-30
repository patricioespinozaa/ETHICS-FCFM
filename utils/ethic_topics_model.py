import spacy
import numpy as np
import json
from utils.bertopic_model import cargar_stopwords, StemmerTokenizer, stop_words_custom
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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


# === Graficar === #
def procesar_y_graficar_topicos(caso, differential):
    # Leer dataframes
    df = pd.read_csv(f"processed_data/{caso}/ETHIC_Topics_df{differential}.csv")

    # Aplicar estilo de Seaborn
    sns.set(style="whitegrid")
    
    # Extraer y contar tópicos para cada etapa y diferencial
    # Expandir las listas en las columnas y convertir a minúsculas
    expanded_topics_ind1 = df['ETHIC_topicos_ind1'].apply(lambda x: eval(x) if isinstance(x, str) else x).explode().str.lower()
    expanded_topics_grup = df['ETHIC_topicos_grup'].apply(lambda x: eval(x) if isinstance(x, str) else x).explode().str.lower()
    expanded_topics_ind2 = df['ETHIC_topicos_ind2'].apply(lambda x: eval(x) if isinstance(x, str) else x).explode().str.lower()

    # Contar las ocurrencias de cada tópico
    topic_counts_ind1 = expanded_topics_ind1.value_counts().reset_index()
    topic_counts_grup = expanded_topics_grup.value_counts().reset_index()
    topic_counts_ind2 = expanded_topics_ind2.value_counts().reset_index()
    
    # Renombrar las columnas
    topic_counts_ind1.columns = ['Tópico', 'Frecuencia']
    topic_counts_grup.columns = ['Tópico', 'Frecuencia']
    topic_counts_ind2.columns = ['Tópico', 'Frecuencia']
    
    # Unir todos los conteos en un solo DataFrame
    all_counts = pd.merge(topic_counts_ind1, topic_counts_grup, on='Tópico', how='outer', suffixes=('_ind1', '_grup'))
    all_counts = pd.merge(all_counts, topic_counts_ind2, on='Tópico', how='outer')
    all_counts.columns = ['Tópico', 'Frecuencia_Ind1', 'Frecuencia_Grup', 'Frecuencia_Ind2']
    
    # Reemplazar NaN por 0
    all_counts.fillna(0, inplace=True)

    # Preparar las frecuencias totales
    all_counts['Frecuencia_Total'] = all_counts['Frecuencia_Ind1'] + all_counts['Frecuencia_Grup'] + all_counts['Frecuencia_Ind2']
    
    # Obtener los tópicos y frecuencias
    top_words = all_counts['Tópico']
    freqs_ind1 = all_counts['Frecuencia_Ind1']
    freqs_grup = all_counts['Frecuencia_Grup']
    freqs_ind2 = all_counts['Frecuencia_Ind2']

    # Posiciones en el eje X
    x = range(len(top_words))

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
    plt.savefig(f"resultados/{caso}/ETHIC_Topics_Dif{differential}.png")
    plt.close()
    
    # Graficar los 10 tópicos más comunes
    topico_frecuencia = all_counts[['Tópico', 'Frecuencia_Total']].sort_values(by='Frecuencia_Total', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(topico_frecuencia['Tópico'], topico_frecuencia['Frecuencia_Total'], color='skyblue')
    plt.xlabel('Frecuencia')
    plt.ylabel('Tópico')
    plt.title(f'Top 10 Tópicos Relevantes Más Comunes - Diferencial {differential}')
    plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar el más común arriba
    plt.tight_layout()
    plt.savefig(f"resultados/{caso}/ETHIC_Top10_Topicos_Dif{differential}.png")
    plt.close()

    # Graficar los 20 tópicos menos comunes
    topico_frecuencia_menor = all_counts[['Tópico', 'Frecuencia_Total']].sort_values(by='Frecuencia_Total').head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(topico_frecuencia_menor['Tópico'], topico_frecuencia_menor['Frecuencia_Total'], color='salmon')
    plt.xlabel('Frecuencia')
    plt.ylabel('Tópico')
    plt.title(f'Top 20 Tópicos Relevantes Menos Comunes - Diferencial {differential}')
    plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar el menos común arriba
    plt.tight_layout()
    plt.savefig(f"resultados/{caso}/ETHIC_Top20_Topicos_Menos_Comunes_Dif{differential}.png")
    plt.close()

def ethic_topics_between_stages(caso):
    print("=== Comparación de Tópicos Éticos entre Etapas ===")
    df1 = pd.read_csv(f"processed_data/{caso}/ETHIC_Topics_df1.csv")
    df2 = pd.read_csv(f"processed_data/{caso}/ETHIC_Topics_df2.csv")
    print("Generando gráfico...")
    # Contar tópicos en Ind1 e Ind2 para df1
    df1['len_ETHIC_topicos_ind1'] = df1['ETHIC_topicos_ind1'].apply(len)
    df1['len_ETHIC_topicos_ind2'] = df1['ETHIC_topicos_ind2'].apply(len)
    df1['topicos_ind2_mayor'] = df1['len_ETHIC_topicos_ind2'] > df1['len_ETHIC_topicos_ind1']
    DF1 = df1['topicos_ind2_mayor'].sum()

    # Contar tópicos en Ind1 e Ind2 para df2
    df2['len_ETHIC_topicos_ind1'] = df2['ETHIC_topicos_ind1'].apply(len)
    df2['len_ETHIC_topicos_ind2'] = df2['ETHIC_topicos_ind2'].apply(len)
    df2['topicos_ind2_mayor'] = df2['len_ETHIC_topicos_ind2'] > df2['len_ETHIC_topicos_ind1']
    DF2 = df2['topicos_ind2_mayor'].sum()

    # Datos para el gráfico
    data = {
        'Diferencial': ['Diferencial 1', 'Diferencial 2'],
        'Frecuencia': [DF1, DF2]
    }
    df_plot = pd.DataFrame(data)

    # Crear el gráfico
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Diferencial', y='Frecuencia', data=df_plot, palette='viridis')

    # Personalizar el gráfico
    plt.title("Estudiantes con más tópicos éticos en Ind2 que en Ind1, por diferencial", fontsize=16)
    plt.xlabel("Diferencial", fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig(f"resultados/{caso}/ETHIC_Topicos_Ind2_Mayor_Ind1.png")
    plt.close()
    print("Gráfico generado exitosamente.")

def ethic_topics_dependency_between_stages(caso):
    print("=== Dependencia de Tópicos Éticos entre Etapas ===")
    df1 = pd.read_csv(f"processed_data/{caso}/ETHIC_Topics_df1.csv")
    df2 = pd.read_csv(f"processed_data/{caso}/ETHIC_Topics_df2.csv")
    print("Generando gráfico...")

    # Asegurarse de que las columnas relevantes son listas
    df1['ETHIC_topicos_ind1'] = df1['ETHIC_topicos_ind1'].apply(eval)  # Suponiendo que las listas están en formato string
    df1['ETHIC_topicos_grup'] = df1['ETHIC_topicos_grup'].apply(eval)  # Suponiendo que las listas están en formato string
    df2['ETHIC_topicos_ind2'] = df2['ETHIC_topicos_ind2'].apply(eval)  # Suponiendo que las listas están en formato string

    # Contar estudiantes que tienen tópicos en Ind2 que no están en Ind1 pero sí en Grup
    def check_dependency(row):
        # Tópicos en Ind2 que no están en Ind1 pero están en Grup
        return any(topic in row['ETHIC_topicos_grup'] and topic not in row['ETHIC_topicos_ind1'] for topic in row['ETHIC_topicos_ind2'])

    df2['tiene_topicos_dep'] = df2.apply(check_dependency, axis=1)
    DF_dependency_count = df2['tiene_topicos_dep'].sum()

    # Datos para el gráfico
    data = {
        'Dependencia': ['Dependencia de Tópicos'],
        'Frecuencia': [DF_dependency_count]
    }
    df_plot = pd.DataFrame(data)

    # Crear el gráfico
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Dependencia', y='Frecuencia', data=df_plot, palette='coolwarm')

    # Personalizar el gráfico
    plt.title("Cantidad de Estudiantes con Tópicos en Ind2 no en Ind1 pero en Grup", fontsize=16)
    plt.xlabel("Dependencia", fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig(f"resultados/{caso}/ETHIC_Topicos_Dependencia_Ind2_no_Ind1_Grup.png")
    plt.close()
    print("Gráfico generado exitosamente.")