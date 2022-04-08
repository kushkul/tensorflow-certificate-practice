# This lab compares accuracy of convolutional and Dense neural network for fashion mnist
# It uses callback to stop training conditional on training accuracy
# It is exactly similar to the graded lab of the same week.

# Using simple network in fashion mnist

import tensorflow as tf


# Getting the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.


# simple network
simple_model = tf.keras.models.Sequential()
simple_model.add(tf.keras.layers.Flatten())
simple_model.add(tf.keras.layers.Dense(128, activation='relu'))
simple_model.add(tf.keras.layers.Dense(10, activation='softmax'))

simple_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Training the simple model')

#simple_model.fit(x_train, y_train, epochs=5)
#test_loss, test_accuracy = simple_model.evaluate(x_test, y_test)

#print('Test accuracy is {}'.format(test_accuracy))

# reshaping input data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape)

# making a convolutional model
conv_model = tf.keras.models.Sequential()
conv_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
conv_model.add(tf.keras.layers.MaxPooling2D((2,2)))
conv_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), activation='relu'))
conv_model.add(tf.keras.layers.MaxPooling2D((2,2)))
conv_model.add(tf.keras.layers.Flatten())
conv_model.add(tf.keras.layers.Dense(128, activation='relu'))
conv_model.add(tf.keras.layers.Dense(10, activation='softmax'))


# Callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.9:
            self.model.stop_training = True

callback_to_use = myCallback()

conv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Training convolutional model')
conv_model.fit(x_train, y_train, epochs=5, callbacks=[callback_to_use])
loss, accuracy = conv_model.evaluate(x_test, y_test)

print('Test accuracy of conv model is {}'.format(accuracy))