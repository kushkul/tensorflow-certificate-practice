import tensorflow as tf


sentences = [
    'I am legend',
    'And I am Kush',
    'Kush is awesome and legend' ]

test_sentences = [
    'Luv and Kush are awesome' ]


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<oov>')

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

print('Work index is')
print(tokenizer.word_index)

print('Sequences for train sentences are')
print(sequences)

test_sequences = tokenizer.texts_to_sequences(test_sentences)

print('Sequences for test sentences are')
print(test_sequences)

# Padding sequences
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

print('Padded sequences are ')
print(padded_sequences)