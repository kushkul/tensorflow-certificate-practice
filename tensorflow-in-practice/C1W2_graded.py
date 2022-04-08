import tensorflow as tf

# Dataset - mnist
# Aim - Create a simple NN and use callback when accuracy is 99%

#loading the data
mnist_data = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist_data.load_data()

x_train = x_train/255.

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


class my_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.1:
            print('\n Loss is less than 0.1 !')
            self.model.stop_training = True


callback_obj = my_callback()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=10, callbacks=[callback_obj])

print(hist)