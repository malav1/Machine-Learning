#IMPORT
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



#IMPORTING THE DATASET
fashion_mnist = keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training




#WHAT THE DATA LOOKS LIKE
print(train_images.shape) #output is "(60000, 28, 28)"- 60k pics of 28px*28px
print(train_images[5,23,23]) #this code prints one px value. px values for these pics will be between 0 and 255, 0 being black and 255 being white. This means its a grayscale pic. if it was colored it would return 3 rgb values.
print(train_labels[:10])  # this code prints the first 10 training labels. Our labels(output) are ints from 0 - 9. Each int represents a specific article of clothing. the list of label names below will indicate which is which.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()
plt.imshow(train_images[30])
plt.colorbar()
plt.grid(False)
# plt.show()




#INPUT DATA PREPROCESSING
train_images = train_images / 255.0
test_images = test_images / 255.0
#here we scale all our px vals(0-255) to be between 0 and 1 by dividing each value in the training and testing sets by 255.0. We do this because smaller values will make it easier for the model to process our values.




#BUILDING THE MODEL
model = keras.Sequential([ #sequential means data passing from the left to the right, through the layers, sequentially
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)- flattens the 28x28 matrix-like structure to a 784px 1D structure
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)-dense layer as stated. 128 is how many neurons we will have. it is a randomly picked number. the activation func we will use, as stated is rectified linear unit
    keras.layers.Dense(10, activation='softmax') # output layer (3)- 10 output neurons (reflecting the number of classes we have as defined by "class_name" list)
])




#COMPILING THE MODEL
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #first 2 elements are self-explanatory. metrics refers to what we want to see form our output.




#TRAINING THE MODEL
model.fit(train_images, train_labels, epochs=1)#fitting it to the training data(another word for training). dont need to do any input funcs at this point bc keras does it all here




#TESTING/EVALUATING THE MODEL
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) #verbose just controls how the output looks on the console as the model is being tested. "test_loss, test_acc" split up the metrics returned to it by model.evaluate
print('Test accuracy:', test_acc)



#MAKING PREDICTIONS
predictions = model.predict(test_images)#passing it a list of images we want predicting. if we wanted to predict only 1 image we could pass it"[test_images[0]]" instead, as a list bc .predict() only accepts lists
print(class_names[np.argmax(predictions[10])]) #argmax() returns the index of the maximium value from a numpy list
plt.figure()
plt.imshow(test_images[10])
plt.colorbar()
plt.grid(False)
plt.show()





#VERIFYING PREDICTIONS
#code below asks user for a number then loads up the index of that image in our model by saying what the image actually is and what the model predicted it to be
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
