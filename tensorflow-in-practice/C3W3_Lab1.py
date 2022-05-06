# Aim - Build a single layer LSTM model using subwords-8k imdb dataset

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Downing the data
data, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_data, test_data = data['train'], data['test']

# Tokenizer of the dataset
tokenizer = info.features['text'].encoder


# Padding and batching the dataset
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EMBEDDING_DIM = 16
LSTM_DIM=8

dataset_train = train_data.shuffle(BUFFER_SIZE)
dataset_train = dataset_train.padded_batch(BATCH_SIZE)
dataset_test = test_data.padded_batch(BATCH_SIZE)

# Building the model
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, EMBEDDING_DIM),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model1.fit(dataset_train, epochs=10, validation_data=dataset_test)

def plot_metrics(history, metrics):
    plt.plot(history.history(metrics))
    plt.plot(history.history('val_'+metrics))
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel(metrics)
    plt.show()

plot_metrics(history, 'accuracy')
plot_metrics(history, 'loss')

