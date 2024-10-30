import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np


def procesar_comentarios(comentarios, tokenizer):
    texto = ' '.join(comentarios.dropna())  
    tokens = tokenizer(texto)  
    return ' '.join(tokens)  

def generar_nube_palabras(caso, texto, titulo, colormap='viridis', background_color='white', max_words=200, mask=None, stop_words_custom=None):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        colormap=colormap,  
        max_words=max_words,
        contour_color='steelblue',  #
        contour_width=1,
        mask=mask, 
        stopwords=STOPWORDS.union(stop_words_custom if stop_words_custom else set()) 
    ).generate(texto)
    
    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(titulo, fontsize=16, color='darkblue')
    plt.tight_layout(pad=0)

    # Save 
    plt.savefig(f"resultados/{caso}/WC_{titulo}.png", dpi=300)
    plt.show()

# Función principal para generar nubes de palabras
def crear_nubes_palabras(caso, df_diferencial_1, df_diferencial_2, tokenizer, stop_words_custom=None, colormap='plasma', background_color='ivory'):
    # Crear nubes de palabras para Diferencial 1
    for columna in ['Comentario - Ind1 - Diferencial 1', 'Comentario - Grup - Diferencial 1', 'Comentario - Ind2 - Diferencial 1']:
        texto_procesado = procesar_comentarios(df_diferencial_1[columna], tokenizer)
        generar_nube_palabras(caso, texto_procesado, f"Nube de Palabras para {columna}", colormap=colormap, background_color=background_color, stop_words_custom=stop_words_custom)

    # Crear nubes de palabras para Diferencial 2
    for columna in ['Comentario - Ind1 - Diferencial 2', 'Comentario - Grup - Diferencial 2', 'Comentario - Ind2 - Diferencial 2']:
        texto_procesado = procesar_comentarios(df_diferencial_2[columna], tokenizer)
        generar_nube_palabras(caso, texto_procesado, f"Nube de Palabras para {columna}", colormap=colormap, background_color=background_color, stop_words_custom=stop_words_custom)
