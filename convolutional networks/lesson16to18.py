#IMPORT
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#CIFAR Image Dataset contains 60,000 32x32 color images with 6000 images of each class.





# IMPORTING AND SPLITTING THE DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() #this line loads the data as an unusual tensorflow object unlike the object types weve been dealing with before which have tended to be NumPy lists which we were able to have a look at (by, e.g. looking at its shape)







#HOW THE DATA LOOKS
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

IMG_INDEX = 7  # change this to look at other images
plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()





# INPUT DATA PROCESSING 
train_images, test_images = train_images / 255.0, test_images / 255.0 #Normalize pixel values to be between 0 and 1







#BUILDING THE MODEL
    #building the CNN (BASE)
#structure is: conv_layer=>max_pool_layer=>conv_layer=>max_pool_layer
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #1st layer=the first number on this line is the amount of filters(the first number on the other conv layers also represent this). the brackets after that contain the sample size (3x3). relu will be applied to the dot product value wich will then be outputted in the response map. 32, 32, 3 means our imgs will be sized 32x32 and have 3 color channels.

model.add(layers.MaxPooling2D((2, 2))) #This layer will perform the max pooling operation using 2x2 samples and a stride of 2.

#The rest of layers do very similar things but take as input the response map from the previous layer. They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks in spacial dimensions as it passed through the layers, meaning we can afford (computationally) to add more depth.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# print(model.summary()) #output is below
'''Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0           30=>15 bc our maxpooling is downsampling by 2  
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
=================================================================
Total params: 56,320
Trainable params: 56,320
Non-trainable params: 0'''
#the output shape for our first layer will be 30, 30, 32 instead of 32,32,32 bc we are not using padding so it is the limit of the number of pxs we can take. the depth of the image also increases but the spacial dimensions reduce drastically with each layer



    #building the dense layer (CLASSIFIER)- the conv base extracts all the features out of the image and we use this dense network to say if this combo of features exist, classify it as this.
model.add(layers.Flatten()) #this line will take the final output of the conv layer(4,4,64) and flatten it to one line/1D
model.add(layers.Dense(64, activation='relu')) #64 neuron dense layer which connects all the flattened data to it with an act func (relu)
model.add(layers.Dense(10)) #output layer of dense 10 neurons (1 neuron for each class)

# print(model.summary()) #output below
''' Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
 flatten (Flatten)           (None, 1024)              0            #calculation of 4x4x64 (being flattened)
                                                                 
 dense (Dense)               (None, 64)                65600     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0'''









#COMPILING THE MODEL
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #to know when to use what optimizer and loss function just look up the most basic for the type of work you want to do on the data








#TRAINING THE MODEL
history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels)) #epochs recc is 10 but bc of time 4 is used. we store the training output in a var to access later








#EVALUATING THE MODEL
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) #outputs the same as the "validation_data" part of the code above.
print(test_acc)


