import tensorflow as tf
import zipfile
import urllib.request
import os

# Downloading the dataset
#!wget https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip
urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip",
                           filename="./data/horse-or-human.zip")

# Unzip the dataset
local_zip = 'data/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('data/horse-or-human')
zip_ref.close()



# Directory with our training horse pictures
train_horse_dir = os.path.join('data/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('data/horse-or-human/humans')


# Getting image dtaa generator
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
train_data_gen = train_gen.flow_from_directory('data/horse-or-human/',
                                               target_size=(300, 300), class_mode='binary',
                                               batch_size=32)

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
valid_data_gen = valid_gen.flow_from_directory('data/horse-or-human/',
                                               target_size=(300, 300), class_mode='binary')


# Defining the mdoel

conv_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

conv_model.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                   metrics=['accuracy'])

history = conv_model.fit(train_data_gen, epochs=5, validation_data=valid_data_gen)

