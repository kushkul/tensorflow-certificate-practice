# Using EMbedding, LSTM, GRU and conv models on imdb reviews dataset

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


data, info = tfds.load('imdb_reviews', as_supervised=True, with_info=True)
data_train = data['train']
data_test = data['test']


train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for s,l in data_train:
    train_sentences.append(s.numpy().decode('utf-8'))
    train_labels.append(l.numpy())

for s,l in data_test:
    test_sentences.append(s.numpy().decode('utf-8'))
    test_labels.append(l.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# Creating tokenizer and processing data set
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<oov>')
tokenizer.fit_on_texts(train_sentences)

train_seq = tokenizer.texts_to_sequences(train_sentences)
test_seq = tokenizer.texts_to_sequences(test_sentences)

max_len = 128
train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_seq, max_len, padding='post', truncating='post')
test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_seq, max_len, padding='post', truncating='post')


def plot_metrics(history1, history2, history3, history4):
    plt.subplot(4,2,1)
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(4, 2, 2)
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(4, 2, 3)
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(4, 2, 4)
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(4, 2, 5)
    plt.plot(history3.history['accuracy'])
    plt.plot(history3.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(4, 2, 6)
    plt.plot(history3.history['loss'])
    plt.plot(history3.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(4, 2, 7)
    plt.plot(history4.history['accuracy'])
    plt.plot(history4.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(4, 2, 8)
    plt.plot(history4.history['loss'])
    plt.plot(history4.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


vocab_size = 10000

# Model 1 - Flatten
flatten_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

flatten_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist1 = flatten_model.fit(x=train_sequences, y=train_labels, epochs=10,
                          validation_data=(test_sequences, test_labels))

# Model 2 - LSTM
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

hist2 = lstm_model.fit(x=train_sequences, y=train_labels, epochs=10,
                       validation_data=(test_sequences, test_labels))


# Model 2 - GRU

gru_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    #tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

hist3 = gru_model.fit(x=train_sequences, y=train_labels, epochs=10,
                      validation_data=(test_sequences, test_labels))


# Model 4 - Convolution

conv_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32, input_length=max_len),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

conv_model.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

hist4 = conv_model.fit(x=train_sequences, y=train_labels, epochs=10,
                       validation_data=(test_sequences, test_labels))


plot_metrics(hist1, hist2, hist3, hist4)