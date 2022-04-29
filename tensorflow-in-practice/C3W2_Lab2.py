# Aim of this lab is to use Sarcasm dataset to train a binary classifier model and get more than 90% of
# accuracy without any spike in the loss function.

import tensorflow as tf
import os
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt


# Downloading the dataset
urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json',
                           filename='data/sarcasm.json')

with open('data/sarcasm.json','r') as json_file:
    sarcasm_data = json.load(json_file)


headlines = []
labels = []
# Reading the data
for item in sarcasm_data:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])

labels = np.array(labels)

# Hyperparameters used in the model
vocab_size = 10000
embedding_dims = 16
maxlen = 50
epoches = 20
padding_choice = 'post'
truncating_choice = 'post'
oov = '<oov>'

# Preparing the sequences for the model
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov)
tokenizer.fit_on_texts(headlines)
sequences = tokenizer.texts_to_sequences(headlines)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen,
                                                                 padding=padding_choice, truncating=truncating_choice)

# Creating the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dims, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print('Training the model')
history = model.fit(padded_sequences, labels, epochs=epoches)

# Plotting the loss and accuracy
print('Plotting the accuracy and loss over time')

print('fail safe')


x=[]
for i in range(0, len(history.history['loss'])):
    x.append(i+1)

plt.plot(x, history.history['loss'] )
plt.plot(x, history.history['accuracy'])
plt.legend(['loss','accuracy'])
plt.show()