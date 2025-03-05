from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from transformers import TrainingArguments, AutoModelForSequenceClassification
import torch
import numpy as np

def CargarModelo(NombreModelo):
    tokenizador = LlamaTokenizer.from_pretrained(NombreModelo)
    modelo = LlamaForCausalLM.from_pretrained(
        NombreModelo, torch_dtype=torch.float16, device_map='auto')
    return(modelo, tokenizador)

def CodificarPrompt(prompt, tokenizador):
    inputIDs= tokenizador(prompt, return_tensors= "pt").input_ids
    return(inputsIDs)

(modelo,tokenizador) = CargarModelo('openlm-research/open_llama_3b_v2')
prompt = 'Q: ¿Cual es el animal mas grande de Chile?\nA:'
inputIDs = CodificarPrompt(prompt, tokenizador)
salida = modelo.generate(input_ids=inputIDs, max_new_tokens=20)

print (tokenizador.decode(salida[0]))
