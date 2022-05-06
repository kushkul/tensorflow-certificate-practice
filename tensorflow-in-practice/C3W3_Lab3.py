# Aim - To build a sequence model made up from convolutional neural network
# using imdb reviews subwords 8k dataset

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


# Downloading the dataset
data, info = tfds.load('imdb_reviews/subwords8k', as_supervised=True, with_info=True)
data_train, data_test = data['train'], data['test']

tokenizer = info.features['text'].encoder


# Preparing the data
BATCH_SIZE = 128
BUFFER = 10000

train_dataset = data_train.shuffle(BUFFER)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = data_test.padded_batch(BATCH_SIZE)


# Verifying the inputs and outputs of the covolution layer
features = 128
FILTERS = 128
KERNEL_SIZE = 5
timesteps = 20
batch_size = 1

random_inp = np.random.rand(batch_size, timesteps, features)
print('Shape of input data array: {}'.format(random_inp.shape))

conv1d = keras.layers.Conv1D(FILTERS, KERNEL_SIZE, activation='relu')
output_data = conv1d(random_inp)
print('Output data shape: {}'.format(output_data.shape))
pool_layer = keras.layers.GlobalAvgPool1D()
pool_out = pool_layer(output_data)
print('Pool layer output shape is : {}'.format(pool_out.shape))

# Building the model

conv_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

conv_model.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

hist = conv_model.fit(train_dataset, epochs=5, validation_data=test_dataset)

def plot_metrics(history):
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

plot_metrics(hist)