# ETHICS-FCFM
Repositorio ayudantía Unidad ETHICS Otoño 2024 | Patricio Espinoza A.

>Las carpetas con datos para la ejecución se encuentran disponibles en el Drive del ayudante.

>En App.ipynb se pueden ejecutar los procesos de análisis así como también solo la generación de los gráficos.

## Casos trabajados:

> - Amanda 2024
> - Amanda 2023
> - Adela 2024
> - Adela 2023
> - Luis 2023

## Análisis realizados:

> **Análisis Gramatical**:
    Permite identificar la cantidad de dependencias gramaticales u oraciones subordinadas en las respuestas de los alumnos por cada etapa y diferencial del caso.

> **Análisis de Tópicos BERT**: 
    Se utiliza el modelo BERT para identificar tópicos (no necesariamente éticos) dentro de las respuestas de los estudiantes.

> **Análisis de Tópicos ETHIC**: 
    Mediante una lista de palabras éticas, se obtienen aquellas presentes en las respuestas de los estudiantes, identificandolas como tópicos.

> **Tópicos distintos entre las etapas**: 
    Contabiliza cuantos tópicos distintos hubieron para las etapas y diferenciales, tanto para el modelo BERT como ETHIC.

> **Comparación de tópicos entre etapas**:
    Para cada estudiante distingue si hubieron tópicos de la etapa grupal presentes en la etapa Ind2, y también si hubieron tópicos grupales presentes en Ind2 que no estaban en la respuesta del estudiante en la etapa Ind1.

> **Nube de palabras**: 
    Luego de un preprocesamiento se observan las palabras más comunes para cada etapa y diferencial.

> **Frecuencia de aparición de palabras éticas**:
    Contabiliza cuantas veces aparecen las palabras éticas dadas a lo largo de todas las respuestas de los estudiantes.

> **Conectores más usados**:
    Se obtiene un tabla con los conectores y su frecuencia para todas las respuestas.

> **Justificaciones post conectores**:
    Generación de una tabla con las palabras post conectores y su relevancia/importancia.