from utils.bertopic_model import cargar_stopwords, StemmerTokenizer, cargar_y_preprocesar_comentarios
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