import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def obtain_comments(directorio_raiz, columnas_dif1, columnas_dif2, ruta_salida):
    """
    Lee archivos Excel en las subcarpetas del directorio dado, selecciona columnas específicas
    de la hoja 'Datos', consolida los datos en un DataFrame y los guarda en un archivo CSV.

    Args:
        directorio_raiz (str): Ruta al directorio que contiene las carpetas con los archivos Excel.
        columnas_dif1 (list): Lista de columnas relacionadas con el diferencial 1.
        columnas_dif2 (list): Lista de columnas relacionadas con el diferencial 2.
        ruta_salida (str): Ruta para guardar el archivo CSV consolidado.

    Returns:
        None
    """
    # Dataframe para guardar todos los comentarios
    df_comentarios = pd.DataFrame()

    print(f"Explorando el directorio: {directorio_raiz}")

    # Recorrer las carpetas en el directorio raíz
    for carpeta in os.listdir(directorio_raiz):
        ruta_carpeta = os.path.join(directorio_raiz, carpeta)
        if os.path.isdir(ruta_carpeta):  # Verifica que sea una carpeta
            print(f"Carpeta detectada: {carpeta}")
            # Recorrer archivos Excel dentro de la carpeta
            for archivo in os.listdir(ruta_carpeta):
                if archivo.endswith(".xlsx"):  # Procesar solo archivos Excel
                    print(f"  Archivo reconocido: {archivo}")
                    ruta_archivo = os.path.join(ruta_carpeta, archivo)
                    try:
                        # Leer el archivo Excel desde la hoja 'Datos'
                        print(f"    Procesando archivo: {archivo}")
                        df = pd.read_excel(ruta_archivo, sheet_name="Datos")
                        columnas_presentes = df.columns.tolist()

                        # Identificar las columnas presentes
                        columnas_interes = []
                        if any(col in columnas_presentes for col in columnas_dif1):
                            columnas_interes.extend([col for col in columnas_dif1 if col in columnas_presentes])
                        if any(col in columnas_presentes for col in columnas_dif2):
                            columnas_interes.extend([col for col in columnas_dif2 if col in columnas_presentes])

                        if columnas_interes:
                            df_filtrado = df[columnas_interes].copy()
                            # Agregar los datos al dataframe acumulador
                            df_comentarios = pd.concat([df_comentarios, df_filtrado], ignore_index=True)
                        else:
                            print(f"    No se encontraron columnas de interés en {archivo}")
                    except Exception as e:
                        print(f"    Error al procesar {archivo}: {e}")

    # Guardar el dataframe consolidado en un nuevo archivo CSV
    df_comentarios.to_csv(ruta_salida, index=False)
    print(f"Archivo consolidado guardado en: {ruta_salida}")

# Aplicación
directorio_raiz = "data/answers"
columnas_comentarios_dif1 = [
    'Comentario - Ind1 - Diferencial 1',
    'Comentario - Grup - Diferencial 1',
    'Comentario - Ind2 - Diferencial 1',
    'Respuesta'
]
columnas_comentarios_dif2 = [
    'Comentario - Ind1 - Diferencial 2',
    'Comentario - Grup - Diferencial 2',
    'Comentario - Ind2 - Diferencial 2'
]
ruta_salida = "processed_data/Training/raw_training_data.csv"

obtain_comments(directorio_raiz, columnas_comentarios_dif1, columnas_comentarios_dif2, ruta_salida)

import os

def stack_columns(input_csv, output_csv, new_column_name="Respuestas"):
    """
    Combina todas las columnas de un archivo CSV en una sola columna,
    eliminando duplicados y valores nulos, y luego borra el archivo original.

    Args:
        input_csv (str): Ruta al archivo CSV de entrada.
        output_csv (str): Ruta al archivo CSV donde se guardará el resultado.
        new_column_name (str): Nombre de la nueva columna unificada.

    Returns:
        None
    """
    try:
        # Leer el archivo CSV
        df = pd.read_csv(input_csv)

        # Apilar todas las columnas en una sola, eliminando NaN
        df_unificada = pd.DataFrame({new_column_name: df.stack().reset_index(drop=True)})

        # Antes de eliminar duplicados, mostrar la cantidad de respuestas
        total_respuestas = df_unificada.shape[0]
        total_duplicados = df_unificada.duplicated().sum()
        total_unicas = total_respuestas - total_duplicados

        print(f"\nTotal de respuestas antes de eliminar duplicados: {total_respuestas}")
        print(f"Respuestas duplicadas encontradas: {total_duplicados}")
        print(f"Respuestas únicas antes de eliminar duplicados: {total_unicas}")

        # Eliminar duplicados y valores nulos
        df_unificada = df_unificada.drop_duplicates().dropna()

        # Después de la eliminación, mostrar la cantidad de respuestas
        total_respuestas_final = df_unificada.shape[0]
        print(f"Total de respuestas después de eliminar duplicados y NaN: {total_respuestas_final}")

        # Guardar el nuevo DataFrame en un archivo CSV
        df_unificada.to_csv(output_csv, index=False)

        # Borrar el archivo original
        os.remove(input_csv)
        print(f"Archivo original {input_csv} eliminado.")

        print(f"Archivo transformado guardado en: {output_csv}")
    except Exception as e:
        print(f"Error al procesar el archivo CSV: {e}")

# Uso de la función
input_csv = "processed_data/Training/raw_training_data.csv"
output_csv = "processed_data/Training/training_data.csv"

stack_columns(input_csv, output_csv)
