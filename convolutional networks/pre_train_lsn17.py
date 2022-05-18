#IMPORT
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()








#DATASET
#This dataset contains (image, label) pairs where images have different dimensions and 3 color channels.
# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))









#DATA REPROCESSING (to make all the imgs the same size)
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32) #we will "cast"(convert) every px in an image to a float bc it could be a dec val
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

#applying the above func to all the imgs
train = raw_train.map(format_example) #map takes the func and applies it to every img in raw_train
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

#how the new imgs look
for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#shuffling and batching the reprocessed imgs
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#comparing old img shape to new one
for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)










#PICKING THE PRE-TRAINED MODEL 
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3) #"3" refers to the number of color channels

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, #here we say if we want to use the classifier that comes with the model. here we will be retraining parts of the network to work for cats and dogs-not for a 1000 diff classes which is what this model was originally trained for
                                               weights='imagenet') # predetermined weights from imagenet(Googles dataset).
# base_model.summary() #shows a table of every CNN layer.

# below, the base_model will output a shape (32, 5, 5, 1280) tensor that is a feature extraction from our original (1, 160, 160, 3) image. The 32 means that the model has detected 32 layers of differnt filters/features.
for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
# print(feature_batch.shape)

#freezing the base
base_model.trainable = False #turning the trainable parameters/attributes of a layer off
# base_model.summary() -if we run this, it will output the same as this line did above, however, it will no say "0" for trainable parameters and say 2mil odd for non-trainable

#adding our classifier to find either cat or dog
#Instead of flattening the feature map of the base layer we will use a global average pooling layer that will average the entire 5x5 area of each 2D feature map (there are 1280 of them) and return to us a single 1D tensor(flattening it)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)










#PUTTING ALL THE LAYERS TOGETHER
model = tf.keras.Sequential([
  base_model, #base
  global_average_layer, #classifier
  prediction_layer #predicition/output
])
# model.summary()
'''_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 mobilenetv2_1.00_160 (Funct  (None, 5, 5, 1280)       2257984    <=base layer.1280 is # of filters/feature maps(depth). 5x5 is the size of each filter/feature map outputted by the filter
 ional)                                                          
                                                                 
 global_average_pooling2d (G  (None, 1280)             0         <=the average of 1280 5x5 feature maps
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 1)                 1281      <=1280 connections from prev. layer +1 bias
                                                                 
=================================================================
Total params: 2,259,265
Trainable params: 1,281
Non-trainable params: 2,257,984
_________________________________________________________________'''










#COMPILING THE MODEL
#adjusting the weights and biases on the layers we are putting on top of the base
base_learning_rate = 0.0001 #how much am i allowed to modify the weights and biases of the network. dont want to make major changes if we dont have to bc of strong base
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
#output is about 50% which confirms its purely guessing






#TRAINING THE MODEL
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future- this method is specific to keras
new_model = tf.keras.models.load_model('dogs_vs_cats.h5') #how to load it