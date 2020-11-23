# A model to test the similarity between handwritten number image and orginal(typed) numbers

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np


#getting the dataset of mnsit from keras and putting it equal to the training and testing model
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


#to scale the dataset
X_train = X_train/255
X_test = X_test/255

# for flatenning the data set i.e to get the array from more than one dimension to other dimension
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

'''
earlier model with only one layer
'''

# model = keras.Sequential([
#     keras.layers.Dense(10, input_shape = (784,), activation='sigmoid')
# ])

# model.compile(
#     optimizer='adam',
#     loss = 'sparse_categorical_crossentropy',
#     metrics = ['accuracy']
# )

'''
New model with more than one layer
'''

model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#fitting the datset and evaluating it

model.fit(X_train_flattened, y_train, epochs = 5)
model.evaluate(X_test_flattened, y_test)

#getting the predicted value

y_predicted = model.predict(X_test_flattened)
y_predicted[0]

#to convert them into the whole number
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Making the confusion matrix
cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
print(cm)










