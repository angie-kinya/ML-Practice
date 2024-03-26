import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#(optional) Disable logging
tf.get_logger().setLevel('ERROR')

#Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Normalize pixel values to the range 0 - 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Reshape the images for suitable training
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

#convert labels to one-hot encoding
train_labels = tf.one_hot(train_labels, depth=10)
test_labels = tf.one_hot(test_labels, depth=10)

plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
plt.show()