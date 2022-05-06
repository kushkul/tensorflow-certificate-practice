# Aim - To use LSTM and Conv network to create RNN on sarcasm dataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request


# Downloading and loading the dataset
SARCASM_DATASET = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'

urllib.request.urlretrieve(SARCASM_DATASET, filename='data/sarcasm.json')
with open('data/sarcasm.json','r') as json_file:
    sarcasm_data = json.load(json_file)

headline = []
label = []
for item in sarcasm_data:
    headline.append(item['headline'])
    label.append(item['is_sarcastic'])


# Constants
VOCAB_SIZE = 10000
EMBEDDING_DIM = 32
OOV = '<oov>'
TRAIN_LEN = 18000
MAXLEN = 128
FILTERS = 128
KERNEL_SIZE = 5


# pre model data preparation
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV)

train_data = headline[:TRAIN_LEN]
test_data = headline[TRAIN_LEN:]
train_label = label[:TRAIN_LEN]
test_label = label[TRAIN_LEN:]

tokenizer.fit_on_texts(train_data)

train_seq = tokenizer.texts_to_sequences(train_data)
test_seq = tokenizer.texts_to_sequences(test_data)

train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=MAXLEN, padding='post', truncating='post')
test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=MAXLEN, padding='post', truncating='post')

train_label = np.array(train_label)
test_label = np.array(test_label)


# Model1 = Bidirectional LSTM model

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


hist1 = lstm_model.fit(x=train_sequences, y=train_label, epochs=10,
                       validation_data=(test_sequences, test_label))


# Mdel2 - conv1D net

conv_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN),
    tf.keras.layers.Conv1D(FILTERS, KERNEL_SIZE, activation='relu'),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

conv_model.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

hist2 = conv_model.fit(x=train_sequences, y=train_label, epochs=10,
                       validation_data=(test_sequences, test_label))


def plot_metrics(hist1, hist2):
    plt.subplot(2,2,1)
    plt.plot(hist1.history['accuracy'])
    plt.plot(hist1.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('LSTM Accuracy')

    plt.subplot(2,2,2)
    plt.plot(hist1.history['loss'])
    plt.plot(hist1.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LSTM Loss')

    plt.subplot(2,2,3)
    plt.plot(hist2.history['accuracy'])
    plt.plot(hist2.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Conv net Accuracy')

    plt.subplot(2,2,4)
    plt.plot(hist2.history['loss'])
    plt.plot(hist2.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Conv net Loss')

    plt.show()

plot_metrics(hist1, hist2)