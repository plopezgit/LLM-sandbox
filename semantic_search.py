import pandas as pd
import numpy as np
import pprint
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from keys import open_ai_key

def BuscarComentarios (Opiniones, consulta, n=3):
    EmbeddingConsulta = get_embedding(
        consulta,
        engine = MODELO_EMBEDDING)
    Opiniones["similitud"] = Opiniones.embedding.apply(lambda x:
                                                       cosine_similarity(x, EmbeddingConsulta))
    results = Opiniones.sort_values("similitud", ascending=False).head(n)
    return results

openai.api_key = open_ai_key
MODELO_EMBEDDING = "text-embedding-ada-002"
Opiniones = pd.read_csv('sentiments-ingles.csv', header=None)
Opiniones["embedding"] = Opiniones[0].apply(lambda x: get_embedding(x, engine=MODELO_EMBEDDING))

results = BuscarComentarios (Opiniones, "borderlands quite", 3)
print(results[0])
    
    
