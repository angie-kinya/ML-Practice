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

#introsduce linear classifier model using Keras sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #flatten the input images
    tf.keras.layers.Dense(10, activation='softmax') #output layer with softmax activation for classification
])

#compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

#evaluate the model using the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

#make predictions of the test data
predictions = model.predict(test_images)

#convert the predictions from probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

#display some predictions with their corresponding true labels
for i in range(10):
    print("Predicted:", predicted_labels[i], "True Label:", np.argmax(test_labels[i]))