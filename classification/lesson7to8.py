from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pandas as pd


#INITIALISING CONSTANTS
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']



#MAKING THE DATARFRAMES/ORGANIZING THE RAW DATA
#keras is a sub-module in tensorflow which has lots of useful datasets and tools
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv") #this line saves the imported data on our device as a .csv
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#the saved data is loaded as a pandas df. the column names will be what we initialised and header will start at index 0
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

#removing the species column as our label as thats what we want our model to predict
train_y = train.pop('Species')
test_y = test.pop('Species')




#INPUT FUNCTION-CONVERSION TO TENSORS
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the raw data to tensors/readable data for the model
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)




#FEATURE COLUMN CREATION 
# Feature columns are used to specify how Tensors received from the input function should be combined and transformed before entering the model.
my_feature_columns = []
#below for loop is the same as looping though CSV_COLUMN_NAMES (minus the species column). but this way is easier. it loops through the already prepped CSV_COLUMN_NAMES on line 26.
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))



#CREATING THE MODEL
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns, #feature columns to tell the model how to receive the data as we did in previous lesson
    hidden_units=[30, 10], #hidden units builds the architecture of the NN using nodes. will be explained better when covering NN
    n_classes=3) #there are 3 classes of flowers for the data to be classified by




#TRAINING THE MODEL
# lambda below basically passes the args to our func
classifier.train( input_fn=lambda: input_fn(train, train_y, training=True), steps=5000) #steps is comparable to epoch. basically says we are going to go through the dataset until 5000 datapoints have been looked at



#TESTING THE MODEL
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result)) #formatting the output so it looks easier to read



#PREDICTION BASED ON USER INPUT
#for converting the user's input to tensor later on(without labels bc we actually want the model to predict the label).
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
user_input_dict = {}

#below will prompt the user to enter a valid numeric measurement for each feature in our features list
print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  user_input_dict[feature] = [float(val)] #each feature is appended to the dict as a key and the number the user inputted is appended as the key's value, in list format

predictions = classifier.predict(input_fn=lambda: input_fn(user_input_dict)) # using user_input_dict as the data point which will get converted to a tensor, in line, via lambda and fed to our model's prediction method.
for dict_element in predictions: #looping through the dict that .predict() has generated as .predict() always returns a dict. this dict contains all the predictions about the user's inputs
    class_id = dict_element['class_ids'][0] #"class ids", at index 0, says which index number from our SPECIES list that we created at the very beginning is the species which best matches the user's input
    probability = dict_element['probabilities'][class_id] #"probabilities" returns a probability for how likely the user's input matches each flower. here we are pulling the probability at the index we got through class_ids to see how closely it is that the user's input actually matches the flower our model has predicted

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))