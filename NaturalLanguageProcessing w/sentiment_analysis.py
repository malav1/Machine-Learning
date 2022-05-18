
#IMPORT
    #This dataset contains 25,000 reviews from IMDB where each one is already preprocessed and has a label as either positive or negative. Each review is encoded by integers that represents how common a word is in the entire dataset. For example, a word encoded by the integer 3 means that it is the 3rd most common word in the dataset.
from secrets import token_urlsafe
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584 #number of unique words

MAXLEN = 250 #max length of a review
BATCH_SIZE = 64 #how many batches we are going to divide the whole dataset into

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# Lets look at one review
# print(train_data[20])







#MORE PREPROCESSING (adding padding)
    #We cannot pass different length data into our neural network. Therefore, we must make each review the same length.
        # if the review is greater than 250 words then trim off the extra words
        # if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)









#CREATING THE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), #going to find a more meaningful representation for those numbers than just their int values via creating vectors for them. 32 is the number of dimensions each vec will have.
    tf.keras.layers.LSTM(32), #we have to tell the LSTM layer that every word will be 32D
    tf.keras.layers.Dense(1, activation="sigmoid") #The stored sentence in the LSTM will then get passed onto this layer for classification as neg or pos. sigmoid is used at the end bc we are predicting the sentiment so if the sentiment is between 0-1 then any num>0.5 we can classify as pos and any num<0.5 as neg. and as we know sigmoid squishes our values between 0-1.
])
# model.summary()
'''_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 32)          2834688     #VOCAB_SIZE*32 dimensions
                                                                 
 lstm (LSTM)                 (None, 32)                8320      
                                                                 
 dense (Dense)               (None, 1)                 33        #output of every dimension+bias
                                                                 
=================================================================
Total params: 2,843,041
Trainable params: 2,843,041
Non-trainable params: 0'''







#COMPILING THE MODEL
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc']) #optimizer isnt super importat here. could use adam again







#TRAINING THE MODEL
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2) #validation split means we will use 20% of the training data to evaluate and validate the model as we go through. the output here showing the validation accuracy shows the model is overfitted bc we are not using enough data so its v.high accuracy from epoch 1







#MAKING PREDICTIONS
    #Since our reviews are encoded we'll need to convert any new review we want to predict into that form so the network can understand it. To do that well load the encodings from the dataset and use them to encode our own data.
word_index = imdb.get_word_index()

    #ENCODE FUNC
def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text) #"text_to_word_sequence" means given some text, convert all that text into "tokens"(individual words)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens] #if the token is in our mapping of 88k words, we will append its int value to the tokens list otherwise we will append 0 to say this char was not in the imdb word bank so we dont know.

  '''tokenz=[]
  for word in tokens:
    if word in word_index:
        tokenz.append(word_index[word]) #appending the index/int that represents the word in the vocab
    else: 
        tokenz.append(0)'''

  return sequence.pad_sequences([tokens], MAXLEN)[0] #pad_sequences works on a list of/multiple sequences so tokens needs to be embedded in another list, returning a list of lists and ofc we return index 0 of this embedded list.

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
# print(encoded), output:
'''[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0  12  17  13  40 477  35 477]'''

   #DECODE FUNC
reverse_word_index = {value: key for (key, value) in word_index.items()} #reversing the word index. the current one goes from word to int whereas we want to flip it to translate a sentence. 

def decode_integers(integers):
    PAD = 0 #if we see 0 it means blank
    text = ""
    for num in integers: #"integers" is our input which will be a list of vals
      if num != PAD: #if its not padding
        text += reverse_word_index[num] + " " #add the lookup of whateever the word for that num is to the text str

    return text[:-1] #return everything except the last space we added
  
# print(decode_integers(encoded)), output: "that movie was just amazing, so amazing"

    #PREDICT FUNC
def predict(text): #text refers to the movie review
  encoded_text = encode_text(text) #encoding the review using encode func 
  pred = np.zeros((1,250)) #blank numpy array, just bunch of 0s, in the shape of (1,250) bc the shape our model expects is (__,250)- 250 being the number of ints/len of movie review we've set
  pred[0] = encoded_text #inserting our one entry into the array weve created
  result = model.predict(pred) 
  print(result[0])

positive_review = "That movie was awesome! really loved it and would watch it again because it was amazingly great"
# predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
# predict(negative_review)
#outputs: [0.7887912]
#         [0.25303566]
