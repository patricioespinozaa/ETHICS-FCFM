import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analizar_dependencias(texto):
    nlp = spacy.load("es_core_news_md")
    doc = nlp(texto)
    
    total_subordinadas = 0
    for sentencia in doc.sents:
        oraciones_subordinadas = [token for token in sentencia if token.dep_ in ("acl", "advcl", "csubj", "ccomp", "xcomp")]
        total_subordinadas += len(oraciones_subordinadas)
    
    return total_subordinadas

def aplicar_dependencias_grupal(df, col_ind1, col_grup, col_ind2):
    # Aplicar analizar_dependencias directamente a col_ind1 y col_ind2 (en todas las filas)
    df['Ind1_d'] = df[col_ind1].apply(lambda x: analizar_dependencias(str(x)))
    df['Ind2_d'] = df[col_ind2].apply(lambda x: analizar_dependencias(str(x)))
    
    grouped = df[['Grup', 'agno', 'seccion', col_grup]].drop_duplicates(subset=['Grup', 'agno', 'seccion'])
    grouped['Grup_d'] = grouped[col_grup].apply(lambda x: analizar_dependencias(str(x)))
    df = df.merge(grouped[['Grup', 'agno', 'seccion', 'Grup_d']], 
                  on=['Grup', 'agno', 'seccion'], 
                  how='left')
    return df

def graficar_boxplot(df1, df2, caso):
    path = f"../resultados/{caso}"

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 5))
    plt.boxplot([
        df1['Ind1_d'], df1['Grup_d'], df1['Ind2_d'],
        df2['Ind1_d'], df2['Grup_d'], df2['Ind2_d']
    ], boxprops=dict(color='black'),
       whiskerprops=dict(color='darkblue'),
       medianprops=dict(color='red'))
    
    plt.xticks([1, 2, 3, 4, 5, 6], ['Ind1_d1', 'Grup_d1', 'Ind2_d1', 'Ind1_d2', 'Grup_d2', 'Ind2_d2'])
    plt.ylabel('Dependencias')
    plt.title('Boxplot de las dependencias encontradas', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"{path}/Analisis_Gramatical_boxplot_dependencias.png", bbox_inches='tight')
    plt.close()

def crear_tabla_dependencias(df1, df2, caso):
    # Crea una tabla con los conteos de dependencias encontradas.
    path = f"../resultados/{caso}"

    diferenciales = []
    for i, df in enumerate([df1, df2], start=1):
        ind1_count = (df['Ind1_d'] < df['Ind2_d']).sum()
        grup_count = (df['Grup_d'] < df['Ind2_d']).sum()
        ind2_count = (df['Ind2_d'] < df['Ind1_d']).sum()
        
        diferenciales.append([f'Diferencial {i}', 'Ind1 < Ind2', ind1_count])
        diferenciales.append([f'Diferencial {i}', 'Grup < Ind2', grup_count])
        diferenciales.append([f'Diferencial {i}', 'Ind2 < Ind1', ind2_count])

    df_table = pd.DataFrame(diferenciales, columns=['Diferencial', 'Comparación', 'Conteo'])
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.savefig(f"{path}/Analisis_Gramatical_tabla_dependencias.png", bbox_inches='tight')
    plt.close()

def graficar_conteos(df1, df2, caso):
    # Genera gráficos de los conteos de dependencias encontradas.
    path = f"../resultados/{caso}"

    diferencial_1_counts = [
        (df1['Ind1_d'] < df1['Ind2_d']).sum(),
        (df1['Grup_d'] < df1['Ind2_d']).sum(),
        (df1['Ind2_d'] < df1['Ind1_d']).sum()
    ]

    diferencial_2_counts = [
        (df2['Ind1_d'] < df2['Ind2_d']).sum(),
        (df2['Grup_d'] < df2['Ind2_d']).sum(),
        (df2['Ind2_d'] < df2['Ind1_d']).sum()
    ]

    bar_width = 0.35
    x = np.arange(len(diferencial_1_counts))

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - bar_width/2, diferencial_1_counts, width=bar_width, label='Diferencial 1', color='lightcoral', edgecolor='black')
    bars2 = plt.bar(x + bar_width/2, diferencial_2_counts, width=bar_width, label='Diferencial 2', color='lightblue', edgecolor='black')

    plt.ylabel('Conteo', fontsize=12)
    plt.title('Conteo de dependencias encontradas', fontsize=14, fontweight='bold')
    plt.xticks(x, ['Ind1<Ind2', 'Grup<Ind2', 'Ind2<Ind1'], fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1 + bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    plt.savefig(f"{path}/Analisis_Gramatical_conteo_dependencias.png", bbox_inches='tight')
    plt.close()

# Función principal que ejecuta el análisis
def ejecutar_analisis(df_diferencial_1, df_diferencial_2, caso):
    df_diferencial_1 = aplicar_dependencias_grupal(df_diferencial_1, 
                                                   'Comentario - Ind1 - Diferencial 1', 
                                                   'Comentario - Grup - Diferencial 1', 
                                                   'Comentario - Ind2 - Diferencial 1')
    
    df_diferencial_2 = aplicar_dependencias_grupal(df_diferencial_2, 
                                                   'Comentario - Ind1 - Diferencial 2', 
                                                   'Comentario - Grup - Diferencial 2', 
                                                   'Comentario - Ind2 - Diferencial 2')
    
    graficar_boxplot(df_diferencial_1, df_diferencial_2, caso)
    crear_tabla_dependencias(df_diferencial_1, df_diferencial_2, caso)
    graficar_conteos(df_diferencial_1, df_diferencial_2, caso)

# Llama a la función principal con tus DataFrames
# ejecutar_analisis(df_diferencial_1, df_diferencial_2)
