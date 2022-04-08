#IMPORT
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

#LOAD THE DATASET-EXPLORE IT-MAKE SURE WE UNDERSTAND IT
#testing the model using new data is important bc if trianing data is re-fed then model could be biased/have memorised all the answers
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
print(dftrain.head())
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
#printing the top 5 rows of the dataframe again but this time with the survival column popped
print(dftrain.head())
#printing the first row of the popped survived column and the first row of the dataframe containing the individual's details. the indexes always correspond even though the surivived column has now been separated.
print(f" they did/did not survive: {y_train.loc[0]}\n",dftrain.loc[0])

#CREATE OUR CATEGORICAL AND NUMERICAL COLUMNS
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

#FOR A LINEAR ESTIMATOR THESE NEED TO BE CREATED AS FEATURE COLUMNS USING THE BELOW ADVANCED SYNTAX
#feature_columns list starts as blank and will go on to store our different feature columns
feature_columns = []
#the for loop below goes through each element of the CATEGORICAL_COLUMNS list
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  #for example, if we print(dftrain["sex"].unique()) the output would be ['male','female']
  #below will create for us a column in the format of numpy array and it will have the feature name we looped through and all the vocab associated with it.
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

#easier for numberic columns. all we need to give is the feature name and dtype. we also ommit .unique() bc with numeric there could be an infinite number of values
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
#example of CATEGORICAL_COLUMNS output: VocabularyListCategoricalColumn(key='sex', vocabulary_list=('male', 'female'), dtype=tf.string, default_value=-1, num_oov_buckets=0)
#example of NUMERIC_COLUMNS output: NumericColumn(key='age', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)