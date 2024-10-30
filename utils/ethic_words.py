from utils.bertopic_model import cargar_stopwords, StemmerTokenizer, cargar_y_preprocesar_comentarios, BERT_contar_topicos_distintos
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Leer las palabras éticas desde un archivo
def read_ethic_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(f.read().splitlines())

# Contar cuantas palabras éticas hay en los comentarios
def contar_palabras_etica(df1, df2, tokenizer):
    palabras_etica = read_ethic_words('dictionaries/ethic_words.txt')
    comentarios = cargar_y_preprocesar_comentarios(df1, df2, tokenizer)

    contador = Counter()
    for comentario in comentarios:
        for palabra in comentario.split():
            if palabra in palabras_etica:
                contador[palabra] += 1
                
    # Convertir el contador en un DataFrame para visualización
    palabras, frecuencias = zip(*contador.items())
    df_frecuencias = pd.DataFrame({'Palabra': palabras, 'Frecuencia': frecuencias})

    # Crear un gráfico bonito
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Frecuencia', y='Palabra', data=df_frecuencias.sort_values(by='Frecuencia', ascending=False), palette='viridis')
    plt.title('Frecuencia de Palabras Éticas en Comentarios', fontsize=16)
    plt.xlabel('Frecuencia', fontsize=14)
    plt.ylabel('Palabras Éticas', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Mostrar el gráfico
    plt.show()
    
    return contador

import ast
# Función para contar tópicos únicos en cada columna ignorando "Sin tópico"
def ETHIC_contar_topicos_unicos(df, columna):
    columna = df[columna].apply(lambda x: ast.literal_eval(x))
    # Filtrar los tópicos únicos que no sean "Sin tópico"
    all_topicos = [topic for topic_list in columna for topic in topic_list if topic != 'Sin tópico']
    return len(set(all_topicos))

import matplotlib.pyplot as plt


# === Topicos distintos entre etapas === #
def distinct_topics(caso):
    # BERT
    df1_BERT = pd.read_csv(f'processed_data/{caso}/BERT_df1.csv')
    df2_BERT = pd.read_csv(f'processed_data/{caso}/BERT_df2.csv')
    # ETHICS
    df1_ETHICS = pd.read_csv(f'processed_data/{caso}/ETHIC_Topics_df1.csv')
    df2_ETHICS = pd.read_csv(f'processed_data/{caso}/ETHIC_Topics_df2.csv')

    # Contar tópicos únicos en cada columna: ETHIC_topicos_ind1	ETHIC_topicos_ind2	ETHIC_topicos_grup
    print('Realizando conteo de tópicos BERT y ETHIC distintos en cada etapa...')
    # Diferencial 1
    BERT_ind1_df1 = BERT_contar_topicos_distintos(df1_BERT, 'BERT_topicos_ind1')
    BERT_grup_df1 = BERT_contar_topicos_distintos(df1_BERT, 'BERT_topicos_grup')
    BERT_ind2_df1 = BERT_contar_topicos_distintos(df1_BERT, 'BERT_topicos_ind2')
    # Diferencial 2
    BERT_ind1_df2 = BERT_contar_topicos_distintos(df2_BERT, 'BERT_topicos_ind1')
    BERT_grup_df2 = BERT_contar_topicos_distintos(df2_BERT, 'BERT_topicos_grup')
    BERT_ind2_df2 = BERT_contar_topicos_distintos(df2_BERT, 'BERT_topicos_ind2')

    # Diferencial 1
    ETHIC_ind1_df1 = ETHIC_contar_topicos_unicos(df1_ETHICS, 'ETHIC_topicos_ind1')
    ETHIC_grup_df1 = ETHIC_contar_topicos_unicos(df1_ETHICS, 'ETHIC_topicos_grup')
    ETHIC_ind2_df1 = ETHIC_contar_topicos_unicos(df1_ETHICS, 'ETHIC_topicos_ind2')
    # Diferencial 2
    ETHIC_ind1_df2 = ETHIC_contar_topicos_unicos(df2_ETHICS, 'ETHIC_topicos_ind1')
    ETHIC_grup_df2 = ETHIC_contar_topicos_unicos(df2_ETHICS, 'ETHIC_topicos_grup')
    ETHIC_ind2_df2 = ETHIC_contar_topicos_unicos(df2_ETHICS, 'ETHIC_topicos_ind2')

    print("Conteo finalizado")

    print("Generación de gráficos...")  
    # Grafico 1, topicos distintos por etapa para BERT
    # Diferencial 1
    plt.figure(figsize=(12, 6))
    plt.bar(['Ind1', 'Grup', 'Ind2'], [BERT_ind1_df1, BERT_grup_df1, BERT_ind2_df1], color=sns.color_palette("Blues")[2])
    plt.title('Tópicos BERT Distintos por Etapa, Diferencial 1', fontsize=16)
    plt.xlabel('Etapa', fontsize=14)
    plt.ylabel('Tópicos Distintos', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'resultados/{caso}/BERT_Distinct_Topics_D1.png')
    plt.close()
    # Diferencial 2
    plt.figure(figsize=(12, 6))
    plt.bar(['Ind1', 'Grup', 'Ind2'], [BERT_ind1_df2, BERT_grup_df2, BERT_ind2_df2], color=sns.color_palette("Blues")[2])
    plt.title('Tópicos BERT Distintos por Etapa, Diferencial 2', fontsize=16)
    plt.xlabel('Etapa', fontsize=14)
    plt.ylabel('Tópicos Distintos', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'resultados/{caso}/BERT_Distinct_Topics_D2.png')
    plt.close()
    # Grafico 2, topicos distintos por etapa para ETHIC 
    # Diferencial 1
    plt.figure(figsize=(12, 6))
    plt.bar(['Ind1', 'Grup', 'Ind2'], [ETHIC_ind1_df1, ETHIC_grup_df1, ETHIC_ind2_df1], color=sns.color_palette("Greens")[2])
    plt.title('Tópicos ETHIC Distintos por Etapa, Diferencial 1', fontsize=16)
    plt.xlabel('Etapa', fontsize=14)
    plt.ylabel('Tópicos Distintos', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'resultados/{caso}/ETHIC_Distinct_Topics_D1.png')
    plt.close()
    # Diferencial 2
    plt.figure(figsize=(12, 6))
    plt.bar(['Ind1', 'Grup', 'Ind2'], [ETHIC_ind1_df2, ETHIC_grup_df2, ETHIC_ind2_df2], color=sns.color_palette("Greens")[2])
    plt.title('Tópicos ETHIC Distintos por Etapa, Diferencial 2', fontsize=16)
    plt.xlabel('Etapa', fontsize=14)
    plt.ylabel('Tópicos Distintos', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'resultados/{caso}/ETHIC_Distinct_Topics_D2.png')
    plt.close()

    print("Gráficos generados")

