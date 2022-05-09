#IMPORT
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


#IMPORTING THE RAW DATA AND REMOVING OUR LABEL (survived)
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')




#CONVERTING THE DATA INTO WHAT OUR TENSORFLOW MODEL WILL ACCEPT (TENSORS)
# below is an input function that converts our current pandas dataframe into a dataset object- the only type of data that the TensorFlow model we are using will accept
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(2000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # returns the function we just made for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)




#CREATING THE FEATURE COLUMNS WHICH TELLS THE MODEL HOW TO USE THE TENSORS 
# copied below from lesson1to4.py to make sure this page runs
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))




#CREATING THE MODEL
#below we also ensure the model is aware of how to process the tensors via the feature columns
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)



#TRAINING THE MODEL
linear_est.train(train_input_fn)  # input function here passes all the tensors to the model



#TESTING THE MODEL
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing using testing data. we store var result bc we will want to look at it later



#RESULTS
clear_output()
print(result['accuracy']) #the result variable is simply a dict of stats about our model. if we print(result) we would get more data about the result in dict format

result = list(linear_est.predict(eval_input_fn)) #.predict() gets the survival probabilities that the model generated
#below we compare the index of the actual data of the people vs the index of what the model predicted. remember the index order never gets messed up.
print(dfeval.loc[4])
print(y_eval.loc[4])
print(result[4]['probabilities'][1])


