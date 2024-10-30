import spacy
import numpy as np

# Cargar el modelo de embeddings de spaCy en español
nlp = spacy.load("es_core_news_md") 

# Diccionario de palabras clave por tópico ético
keywords_dict = {
    'dignidad': ['dignidad', 'respeto', 'honor', 'valía', 'estima'],
    'bienestar': ['bienestar', 'salud', 'felicidad', 'calidad de vida', 'prosperidad'],
    'justicia': ['justicia', 'injusticia', 'equidad', 'igualdad', 'imparcialidad'],
    'moralidad': ['moralidad', 'moral', 'ética', 'integridad', 'rectitud'],
    'responsabilidad': ['responsabilidad', 'deber', 'obligación', 'compromiso'],
    'honestidad': ['honestidad', 'sinceridad', 'transparencia', 'franqueza', 'veracidad'],
    'autonomía': ['autonomía', 'independencia', 'libertad', 'autodeterminación'],
    'beneficencia': ['beneficencia', 'bienestar', 'altruismo', 'solidaridad'],
    'no maleficencia': ['no maleficencia', 'no dañar', 'prevención del daño'],
    'confidencialidad': ['confidencialidad', 'privacidad', 'secreto', 'protección de datos'],
    'lealtad': ['lealtad', 'fidelidad', 'compromiso', 'confianza'],
    'tolerancia': ['tolerancia', 'aceptación', 'respeto a la diversidad', 'comprensión'],
    'desigualdad': ['desigualdad', 'inequidad', 'discriminación', 'injusticia social', 'exclusión'],
    'educación': ['educación', 'formación', 'enseñanza', 'aprendizaje']
}

def ethic_palabras_clave_en_comentario(comentario):
    """Encuentra las palabras clave en el comentario y asigna un peso basado en su aparición."""
    ponderaciones = {}
    
    # Procesar el comentario
    doc = nlp(comentario.lower())
    
    # Crear un conjunto de stopwords
    stopwords = nlp.Defaults.stop_words
    
    for token in doc:
        if token.is_alpha and token.text not in stopwords:  # Solo palabras y no stopwords
            for key, values in keywords_dict.items():
                # Comprobar la similitud semántica
                if token.lemma_ in values or token.lemma_ == key:
                    ponderaciones[key] = ponderaciones.get(key, 0) + 1  # Aumentar peso si la palabra está en los valores o es la clave

                # Comprobar si hay palabras similares en el diccionario
                for word in values:
                    if token.similarity(nlp(word)) > 0.7:  # Umbral de similitud
                        ponderaciones[key] = ponderaciones.get(key, 0) + 1

    return ponderaciones

def predecir_topicos_eticos(comentario):
    """Predice los tópicos éticos identificados en un comentario."""
    ponderaciones = ethic_palabras_clave_en_comentario(comentario)
    return [topic for topic, peso in ponderaciones.items() if peso > 0]  # Solo incluir tópicos con peso positivo

# Ejemplo de uso:
comentario = "La dignidad humana es fundamental en la ética y debemos fomentar el bienestar de todos."
topicos_predichos = predecir_topicos_eticos(comentario)
print("Tópicos éticos identificados:", topicos_predichos)
