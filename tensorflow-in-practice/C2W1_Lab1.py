import tensorflow as tf
import urllib.request
import zipfile
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Downloading the dataset
print('Downloading the dataset')
#urllib.request.urlretrieve(url='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
#                           filename='data/cats_and_dogs_filtered.zip')

# Unzipping the files
print('Unzipping the dataset')
#with zipfile.ZipFile('data/cats_and_dogs_filtered.zip', 'r') as zip_ref:
#    zip_ref.extractall('data/cats_and_dogs_filtered')

train_cat_images = os.listdir('data/cats_and_dogs_filtered/train/cats/')
train_dog_images = os.listdir('data/cats_and_dogs_filtered/train/dogs/')

test_cat_images = os.listdir('data/cats_and_dogs_filtered/validation/cats/')
test_dog_images = os.listdir('data/cats_and_dogs_filtered/validation/dogs/')

print('Total train Cat images = {}'.format(len(train_cat_images)))
print('Total train dog images = {}'.format(len(train_dog_images)))

print('Total test Cat images = {}'.format(len(test_cat_images)))
print('Total test dog images = {}'.format(len(test_dog_images)))

print('Loading one image')
# Loading one image from train dataset
fig = plt.gcf()
#fig.set_size_inches(1*4, 1*4)
base_path = './data/cats_and_dogs_filtered/train/cats/'
image_path = os.path.join(base_path, train_cat_images[1])

img = mpimg.imread(image_path)
#plt.imshow(img)
#plt.show()

# Checking the maximum value of the pixel, helpful for rescalling and # Checking the dimensions of the image
print(type(img))
#print(img)
print(img.shape) # Every image has different dimension
print(np.max(img)) # Max pixel is 255

# getting image data generator
train_basepath = 'data/cats_and_dogs_filtered/train/'
valid_basepath = 'data/cats_and_dogs_filtered/validation/'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow_from_directory(train_basepath,target_size=(180, 180),
                                              class_mode='binary')
valid_gen = valid_datagen.flow_from_directory(valid_basepath, target_size=(180, 180),
                                              class_mode='binary')


# Building the model
conv_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(conv_model.summary())

# Defining the call back
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy')>0.8:
            print('Accuracy reached 0.8, cancelling training')
            self.model.stop_training=True


callback = myCallBack()

# Setting up th e training
conv_model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                   loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
hist = conv_model.fit(train_gen, epochs=10,
                      callbacks=[callback], validation_data=valid_gen)


