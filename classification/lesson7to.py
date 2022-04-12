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
print(train.head())




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
print(my_feature_columns)