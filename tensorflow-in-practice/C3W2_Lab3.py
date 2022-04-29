# Aim of this script is to understand the difference between the simple text and subwords
# tokenizer of imdb data set and build a binary classifier using the subwords dataset.

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
tf.keras.losses.

# 1. Difference between plaintext and subwords dataset
# Loading the text dataset
data_plaintext, info_plaintext = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
plaintext_train, plaintext_test = data_plaintext['train'], data_plaintext['test']

# Loading the subwords dataset
data_subwords, info_subwords = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
subwords_train, subwords_test = data_subwords['train'], data_subwords['test']


# Checking out the outputs of both datasets
print(info_plaintext.features['text'])

# Printing examples of plaintext dataset
for sentence, label in data_plaintext['train'].take(2):
    print(sentence.numpy())

sentences_train = []
for s, _ in plaintext_train:
    sentences_train.append(s.numpy().decode('utf-8'))
sentences_test = []
for s,_ in plaintext_test:
    sentences_test.append(s.numpy().decode('utf-8'))

print(info_subwords.features['text'])
for example in subwords_train.take(2):
    print(example)

# Observation - the subwords dataset train data are actually small tokens
# while the plain dataset training data are english sentences

# We can use the encoder (tokenizer) used in the subwords dataset to get back the english sentences
print('Printing sentences from subwords dataset')
tokenizer_subwords = info_subwords.features['text'].encoder

for example in data_subwords['train'].take(2):
    print(tokenizer_subwords.decode(example[0]))


# number of tokens = number of words in the plaintext dataset, check below -

sentences = []
for sen, label in data_plaintext['train']:
    sentences.append(sen.numpy().decode('utf-8'))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print('Sentence and its tokenized sequence is below')

print(sentences[0])
print(sequences[0])

print('Converting back the tokens into the sentences')
print(tokenizer.sequences_to_texts(sequences[0:1]))

# Observation - There are a lot of <oov> in the text converted back from the sequence because
# its very difficult to have a very large word index.
# If we have to avoid having very large word index and still dont want to have <oov> words thats when subwords dataset are helpful
# How? They have small words as tokens so that big words could be created using those.
# Subword text encoding gets around this problem by using parts of the word to compose whole words. This makes it
# more flexible when it encounters uncommon words.

print('Subwords tokenizer vocab')
print(tokenizer_subwords.subwords)

# In general, subwords tokenizer works better for new data
my_sentence = 'Tensorflow, from basics to advanced'

sequence_plain = tokenizer.texts_to_sequences([my_sentence])
sentence_plain = tokenizer.sequences_to_texts(sequence_plain)
print('Coverted sentence using plain tokenizer is: ')
print(sentence_plain)

sequence_sub = tokenizer_subwords.encode(my_sentence)
sentence_sub = tokenizer_subwords.decode(sequence_sub)
print('Converted sentence using subwords tokenizer is :')
print(sentence_sub)


# 2. Training the model on tokenized dataset

# We dont have to train or fit the tokenizer, it is already done for us.
# We just have to fix the padding and divide the dta into batches

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_data, test_data = data_subwords['train'], data_subwords['test']
train_dataset = train_data.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

subwords_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer_subwords.vocab_size, 64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(subwords_model.summary)
subwords_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training
hist = subwords_model.fit(train_dataset, epochs=10, validation_data=test_dataset)


# Plotting the loss and the accuracy
def plot_metrics(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.legend()
    plt.show()

plot_metrics(hist, 'accuracy')
plot_metrics(hist, 'loss')

