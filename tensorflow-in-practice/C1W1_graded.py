import tensorflow as tf
import numpy as np


# Dataset - hypothetical house pricing
# Aim - Create 1 layer 1neuron NN to predict house prices

#-----------------------------------
# A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1 bedroom
# house cost 100k, a 2 bedroom house cost 150k etc.
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as
# costing close to 400k etc.
#-----------------------------------

x = [1., 2., 3., 4., 5., 6.]
y = [1., 1.5, 2., 2.5, 3., 3.5]


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, activation='relu', input_shape=(1,)))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, epochs=1000)

print(model.predict([7.]))