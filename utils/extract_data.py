import pandas as pd
import os 
import warnings

# Ignorar advertencias específicas de openpyxl
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def process_data():
    print("Casos disponibles:")
    print("1. Amanda")
    try:
        caso = int(input("Ingresar número del caso :"))
    except ValueError:
        print("Por favor ingrese un número válido.")
        return

    casos_disponibles = {1: "Amanda"}
    
    # Chequeo de opciones
    if caso not in casos_disponibles.keys():
        print("Número de caso no válido")
        return
    
    # Seleccionar caso
    caso_seleccionado = casos_disponibles[caso]
    print(f"Caso seleccionado: {caso_seleccionado}")

    # PATH a las carpetas de datos
    ANSWERS_DATA_PATH = 'data/answers/Amanda'
    FOLDERS_ANSWERS = os.listdir(ANSWERS_DATA_PATH)

    # Dataframe para almacenar los datos
    df_answers = pd.DataFrame(columns=['df', 'opt_left', 'Grup', 'Ind1', 'Ind2', 'Magnitud_Ind1_Grup', 'Magnitud_Grup_Ind2', 
                                    'Magnitud_Ind1_Ind2', 'Cambio_postura_Ind1_Grup', 'Cambio_postura_Grup_Ind2', 
                                    'Cambio_postura_Ind1_Ind2', 'Nivel_Ind1_Grup', 'Nivel_Grup_Ind2', 'Nivel_Ind1_Ind2', 
                                    'Direccion_Ind1_Grup', 'Direccion_Grup_Ind2', 'Direccion_Ind1_Ind2', 
                                    'Comentario - Ind1 - Diferencial 1', 'Comentario - Ind1 - Diferencial 2', 
                                    'Comentario - Grup - Diferencial 1', 'Comentario - Grup - Diferencial 2', 
                                    'Comentario - Ind2 - Diferencial 1', 'Comentario - Ind2 - Diferencial 2', 'agno', 'seccion'])

    # Leer archivos excel del directorio y extraer agno y seccion
    for file in FOLDERS_ANSWERS:
        parts = file.split('_')  
        if len(parts) >= 3: 
            agno = parts[1]  
            seccion = parts[-1][:-5] 
            print(f"Archivo: {file}, Año: {agno}, Sección: {seccion}")
        else:
            print(f"==== Error en extracción de año y sección para el archivo: {file} ====")
            agno = None
            seccion = None
        
        # Extracción de columnas
        df = pd.read_excel(f'{ANSWERS_DATA_PATH}/{file}', sheet_name='Datos')
        df = df[['df', 'opt_left', 'Grup', 'Ind1', 'Ind2', 'Magnitud_Ind1_Grup', 'Magnitud_Grup_Ind2', 
                'Magnitud_Ind1_Ind2', 'Cambio_postura_Ind1_Grup', 'Cambio_postura_Grup_Ind2', 
                'Cambio_postura_Ind1_Ind2', 'Nivel_Ind1_Grup', 'Nivel_Grup_Ind2', 'Nivel_Ind1_Ind2', 
                'Direccion_Ind1_Grup', 'Direccion_Grup_Ind2', 'Direccion_Ind1_Ind2', 
                'Comentario - Ind1 - Diferencial 1', 'Comentario - Ind1 - Diferencial 2', 
                'Comentario - Grup - Diferencial 1', 'Comentario - Grup - Diferencial 2', 
                'Comentario - Ind2 - Diferencial 1', 'Comentario - Ind2 - Diferencial 2']]
        
        # Añadir columnas de agno y seccion
        df['agno'] = agno
        df['seccion'] = seccion
        
        df_answers = pd.concat([df_answers, df], ignore_index=True)
    
    # Sort, fillna y convertir columnas a string
    df_answers.sort_values(by=['Grup', 'seccion', 'agno'], inplace=True)
    df_answers.fillna('', inplace=True)

    # Convertir las columnas especificadas a tipo string
    columnas_a_convertir = [
        'Comentario - Ind1 - Diferencial 1',
        'Comentario - Grup - Diferencial 1',
        'Comentario - Ind2 - Diferencial 1',
        'Comentario - Ind1 - Diferencial 2',
        'Comentario - Grup - Diferencial 2',
        'Comentario - Ind2 - Diferencial 2'
    ]

    # Convertir las columnas a string
    for columna in columnas_a_convertir:
        df_answers[columna] = df_answers[columna].astype(str)

    # Ruta para almacenar los datos procesados
    folder_path = f'processed_data/{caso_seleccionado}'

    # Verificar si la carpeta existe, y si no, crearla
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"\nCarpeta creada: {folder_path}")
    else:
        print(f"\nCarpeta ya existe: {folder_path}")

    # Guardar datos como un nuevo csv en una ruta
    df_answers.to_csv(f'processed_data/{caso_seleccionado}/answers_by_secc_{caso_seleccionado}.csv', index=False)
    print(f"Datos procesados guardados en: {folder_path}/answers_by_secc_{caso_seleccionado}.csv")

    return caso_seleccionado


def read_data(caso):
    # Leer datos procesados
    df = pd.read_csv(f'processed_data/{caso}/answers_by_secc_{caso}.csv')
    print(f"\nDatos leídos de: processed_data/{caso}/answers_by_secc_{caso}.csv")

    # diferencial 1 conserva las columnas: 'Grup', 'Comentario - Ind1 - Diferencial 1', 'Comentario - Grup - Diferencial 1', 'Comentario - Ind2 - Diferencial 1' para df=1
    df1 = df[df['df'] == 1][['Grup', 
                                    'Comentario - Ind1 - Diferencial 1', 
                                    'Comentario - Grup - Diferencial 1', 
                                    'Comentario - Ind2 - Diferencial 1', 'agno', 'seccion']]

    # diferencial 2 conserva las columnas: 'Grup', 'Comentario - Ind1 - Diferencial 2', 'Comentario - Grup - Diferencial 2', 'Comentario - Ind2 - Diferencial 2' para df=2
    df2 = df[df['df'] == 2][['Grup', 
                                    'Comentario - Ind1 - Diferencial 2', 
                                    'Comentario - Grup - Diferencial 2', 
                                    'Comentario - Ind2 - Diferencial 2', 'agno', 'seccion']]

    # Distinguir por diferenciales 
    return df1, df2