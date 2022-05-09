#IMPORT
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#IMPORTING IMAGES TO BE USED
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# CREATING AN IMAGEDATAGENERATOR OBJECT/INSTANCE (which will transform the images)
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

#PREPPING IMGS FOR CONVERSION
test_img = train_images[20] # pick an image to transform
img = image.img_to_array(test_img)  # convert image from weird dataset object to numpy array
img = img.reshape((1,) + img.shape)  # reshapes image to start with "1" continued by whatever the shape of the array is 



#CONVERSION
#the loop below runs forever until we break, saving images to current directory with specified prefix
i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  #save_prefix and format mean every new image will be saved with test in the name followed by something else so, e.g., test1.jpeg,test2.jpeg,test3.jpeg.... 
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0])) #shows us each result- not working here bc code needs to be in conjunction with main .py file
    i += 1
    if i > 4:  # limits augs to 4 per img
        break

plt.show()