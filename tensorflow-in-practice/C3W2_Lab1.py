# Binary classifier on IMDB Reviews dataset

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time

# Downloading the data
data, info = tfds.load('imdb_reviews', as_supervised=True, with_info=True)
train_data, test_data = data['train'], data['test']


# Doing nlp stuff with the data - converting the sentences into sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token='<oov>')

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for s,l in train_data:
    train_sentences.append(s.numpy())
    train_labels.append(l.numpy())

for s,l in test_data:
    test_sentences.append(s.numpy())
    test_labels.append(l.numpy())

# For errors - TypeError: a bytes-like object is required, not 'dict'
train_sentences = [w.decode('utf-8') for w in train_sentences]
test_sentences = [w.decode('utf-8') for w in test_sentences]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

tokenizer.fit_on_texts(train_sentences)
sequences = tokenizer.texts_to_sequences(train_sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=120, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,
                                                                      maxlen=120, padding='post')


# Building the neural network

print('Creating model using flatten layer')
flatten_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(20000, 16, input_length=120),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

flatten_model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])


print('Creating model using Global Average layer')

avg_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(20000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

avg_model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

print('Training flatten model')

time_start = time.time()
#flatten_model.fit(padded_sequences, train_labels, epochs=10,
#                  validation_data=(test_padded_sequences, test_labels))

time_end = time.time()
print('Time taken to train flatten model is {}'.format(time_end-time_start))


print('Training nlp model with global average pooling layer')
time_start_nlp = time.time()
avg_model.fit(padded_sequences, train_labels, epochs=10,
              validation_data=(test_padded_sequences, test_labels))

time_end_nlp = time.time()
print('Time taken to train global average model is {}'.format(time_end_nlp-time_start_nlp))


# Saving the embedding weights to files for analysis

embedding_layer = avg_model.layers[0]
weights = embedding_layer.get_weights()
# print('SHape of embedding weights is: {}'.format(weights.shape))

index_word = tokenizer.index_word

import io

# Open writeable files
out_v = io.open('./data/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('./data/meta.tsv', 'w', encoding='utf-8')

for i in range(1, 20000):
    word_weight = weights[0][i]
    word = index_word[i]
    out_v.write('\t'.join([str(x) for x in word_weight])+'\n')
    out_m.write(word + '\n')

out_v.close()
out_m.close()
