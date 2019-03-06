# Import packages and set numpy random seed
import numpy as np
np.random.seed(5) 
import tensorflow as tf
tf.set_random_seed(2)
from datasets import sign_language
import matplotlib.pyplot as plt
%matplotlib inline

# Load pre-shuffled training and test datasets
(x_train, y_train), (x_test, y_test) = sign_language.load_data()

#Now we'll begin by creating a list of string-valued labels containing the letters that appear in the dataset.
#Then, we visualize the first several images in the training data, along with their corresponding labels.
# Store labels of dataset
labels = ['A','B','C']

# Print the first several training images, along with the labels
fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    ax.set_title("{}".format(labels[y_train[i]]))
plt.show()

#Let's examine how many images of each letter can be found in the dataset.
#Each entry in y_train and y_test is one of 0, 1, or 2, corresponding to the letters 'A', 'B', and 'C', respectively.
#We will use the arrays y_train and y_test to verify that both the training and test sets each have roughly equal proportions of each letter.

# Number of A's in the training dataset
num_A_train = sum(y_train==0)
# Number of B's in the training dataset
num_B_train = sum(y_train==1)
# Number of C's in the training dataset
num_C_train = sum(y_train==2)

# Number of A's in the test dataset
num_A_test = sum(y_test==0)
# Number of B's in the test dataset
num_B_test = sum(y_test==1)
# Number of C's in the test dataset
num_C_test = sum(y_test==2)

# Print statistics about the dataset
print("Training set:")
print("\tA: {}, B: {}, C: {}".format(num_A_train, num_B_train, num_C_train))
print("Test set:")
print("\tA: {}, B: {}, C: {}".format(num_A_test, num_B_test, num_C_test))

#OneHotEncoder
from keras.utils import np_utils
from keras.utils import to_categorical
# One-hot encode the training labels
y_train_OH = to_categorical(y_train)

# One-hot encode the test labels
y_test_OH = to_categorical(y_test)

#Designing the model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential

model = Sequential()
# First convolutional layer accepts image input
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(50, 50, 3)))
# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(4,4)))   
# Add a convolutional layer
model.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu', input_shape=(50, 50, 3)))
# Add another max pooling layer
model.add(MaxPooling2D(pool_size=(4,4)))   
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Summarize the model
model.summary()
# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Obtain accuracy on test set
score = model.evaluate(x=x_test, y=y_test_OH, verbose=0)
print('Test accuracy:', score[1])

