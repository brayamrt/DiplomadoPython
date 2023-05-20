# Se importan las librerias para el template y los renders
from django.shortcuts import render


# Librerias de la Red Neuronal
import numpy as np
import pandas as pd
# Librerias para gráficas
import matplotlib.pyplot as plt
import seaborn as sns

# Libreria para dividir los datos
from sklearn.model_selection import train_test_split
# Libreria para métricas
from sklearn.metrics import confusion_matrix, classification_report
# Framework Tensorflow
import tensorflow as tf


# -----------------------------------

def main(request):
    # *** Plantilla ***
    return render(request, 'index.html', context={})


def prediccion(request):
    
    # Cargue de los datos
    data1 = pd.read_csv('/Users/josellanos/Documents/GitHub/JMLM/Django/proyecto2/proyecto2/dataset/Youtube01-Psy.csv')
    data2 = pd.read_csv('/Users/josellanos/Documents/GitHub/JMLM/Django/proyecto2/proyecto2/dataset/Youtube02-KatyPerry.csv')
    data3 = pd.read_csv('/Users/josellanos/Documents/GitHub/JMLM/Django/proyecto2/proyecto2/dataset/Youtube03-LMFAO.csv')
    data4 = pd.read_csv('/Users/josellanos/Documents/GitHub/JMLM/Django/proyecto2/proyecto2/dataset/Youtube04-Eminem.csv')
    data5 = pd.read_csv('/Users/josellanos/Documents/GitHub/JMLM/Django/proyecto2/proyecto2/dataset/Youtube05-Shakira.csv')

    # Se concatenan los archivos para generar el CORPUS
    data = pd.concat([data1, data2, data3, data4, data5])
    data.drop_duplicates()
    data.reset_index()

    # Se extrae la longitud de vocabulario y la secuencia
    x_train, x_test, y_train, y_test = preprocessing(data)

    # Definición del modelo
    inputs = tf.keras.Input(227,)
    x = tf.keras.layers.Embedding(input_dim = 3821, output_dim = 300)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy', tf.keras.metrics.AUC(name = 'auc')]
    )

    # Entrenamiento de la Red Neuronal
    history = model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        batch_size = 32,
        epochs = 100,
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss',
                patience = 5,
                restore_best_weights = True
            )
        ]
    )

    # Predicción
    response = model.predict(x_test)

    contador = 0
    prediccion = []

    for i in response:
        # Se utilizan los primeros 50 registros de prueba
        if contador < 50:
            # Se obtiene el porcentaje de la predicción
            prediccion.append(i)
            contador = contador + 1


    # El context debe ser un diccionario
    context = {'prediccion': prediccion}

    # *** Plantilla ***
    return render(request, 'index.html', context=context)




def get_sequences(texts, tokenizer, train= True, max_seq_len = None):
    sequence = tokenizer.texts_to_sequences(texts)
    if train == True:
        max_seq_len = np.max(list(map(len, sequence)))
    
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, 
                                                              maxlen = max_seq_len, 
                                                              padding = 'post')
    return sequence





def preprocessing(df):
    df = df.copy()
    X = df['CONTENT']
    Y = df['CLASS']
    # Division de los datos: 70% entrenamiento, 30% pruebas
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size = 0.30,
                                                        shuffle = True,
                                                        random_state = 1)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x_train)
    x_train = get_sequences(texts = x_train, tokenizer = tokenizer, train = True)
    x_test = get_sequences(texts = x_test, tokenizer = tokenizer, train = False, max_seq_len = x_train.shape[1])
    print(f'Longitud Vocabulario: {len(tokenizer.word_index)+1}')
    print(f'Longitud Secuencia: {x_train.shape[1]}')
    return x_train, x_test, y_train, y_test