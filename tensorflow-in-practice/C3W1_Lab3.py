# Aim of this lab is to use sarcasm dataset and create padded sequences of its headline

import tensorflow as tf
import urllib.request
import json

# Downloading the dataset
url = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'
urllib.request.urlretrieve(url, filename='data/sarcasm.json')

# Reading the data
with open('data/sarcasm.json','r') as file:
    json_data = json.load(file)

headlines = []
labels = []
links = []

for item in json_data:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])
    links.append(item['article_link'])


# Using tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=500, oov_token='<oov>')
tokenizer.fit_on_texts(headlines)
sequences = tokenizer.texts_to_sequences(headlines)

# Padd the sequences
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

