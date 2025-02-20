from transformers import InputExample, InputFeatures
from bert.tokenization.bert_tokenization import FullTokenizer
from bert import tokenization
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def CargarBert():
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
    bert_layer = hub.KerasLayer(module_url, trainable=False)
    return(bert_layer)

def CargarTokenizador():
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)
    return(tokenizer)

def bert_encoder(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    for text in texts:
        text = tokenizer.tokenize(text)
        text_sequence = text[:max_len-2]
        input_sequences = ["[CLS]"] + text_sequence + ["[SEP]"]
        pad_len = max_len - len(input_sequences)
        tokens = tokenizer.convert_tokens_to_ids(input_sequences)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequences) + [0] * pad_len
        segment_ids = [0] * max_len
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def ContruirModelo(bert_layer, num_clases, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype= tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype= tf.int32, name="segment_ids")

    _, salida_secuencia = bert_layer([input_word_ids, input_mask, segment_ids])

    salida_clasificador = salida_secuencia[:, 0, :]

    capa_salida = Dense(num_clases, activation='sigmoid')(salida_clasificador)
    
    modelo = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=capa_salida)

    modelo.compile(Adam(learning_rate=2e-6),
                   loss= 'binary_crossentropy',
                   metrics=['accuracy'])
    return modelo

bert_layer = CargarBert()
tokenizer = CargarTokenizador()

opiniones = pd.read_csv('sentiments-ingles.csv', header=None)
opiniones[1].replace(['Negative', 'Positive'], [0,1], inplace=True)

train, test = train_test_split(opiniones)

train_input = bert_encoder(train[0], tokenizer, max_len=160)

modelo_final = ContruirModelo(bert_layer, 1, 160)


#Entrenamiento

#train_history = modelo_final.fit(
#  train_input, train[1],
#  validation_split=0.2,
#  epochs=3,
#  batch_size=16
#  )

#Prueba y validacion

test_input= bert_encoder(test[0], tokenizer, max_len=160)
ProbPrediccion = modelo_final.predict(test_input)
prediccion = np.where(ProbPrediccion>.5, 1, 0)
prediccion


