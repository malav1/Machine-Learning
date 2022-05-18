#IMPORTS
    #We'll write a play using a character predictive model that will take as input a variable length sequence and predict the next character. We can use the model many times in a row with the output from the last predicition as the input for the next call to generate a sequence.
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np





#DATASET
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') #downloading the file. utils allows us to get a file and save it as x

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') #open in read bytes mode. will read it in as an entire string and will be decoded in utf-8 format
print ('Length of text: {} characters'.format(len(text))) # length of text is the number of characters in it

# Take a look at the first 250 characters in text
print(text[:250])







#PREPROCESSING (encoding them as ints)
    #each unique character will be encoded as a different integer. this is easier bc there is a finite number of chars vs words. also chars arent meaningful so we dont need to worry about their vector positioning
vocab = sorted(set(text)) #sort all the unique chars in the text (these include symbols, lower case, upper case, etc.)

char2idx = {u:i for i, u in enumerate(vocab)} # Creating a mapping from unique characters to indices. this line stores as a dict "whatever the string is:0" for every char in vocab
idx2char = np.array(vocab) #turning vocab into an array so we can use the index at which a letter appears as the reverse mapping. index=>letter rather than letter=>index

def text_to_int(text): #func taking a text and returning the int representation for each char in the text
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text) #converting the entire loaded file from above

# lets look at how part of our text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

    #DECODING
def int_to_text(ints):
  try:
    ints = ints.numpy() #input needs to be an array. try to conv to numpy array but if already numpy array continue
  except:
    pass
  return ''.join(idx2char[ints]) #replaces ints with the letters found at those indexes in idx2char array and joins them

print(int_to_text(text_as_int[:13]))








#TRAINING
    # our task is to feed the model a sequence and have it return to us the next character. This means we need to split our text data from above into many shorter sequences that we can pass to the model as training examples.
    # The training examples we will prepare will be texts of (x length) as input and texts of (x length- take away the first char of the text and predict +1 char at the end of the text) as output. it will be shifted one letter to the right
seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1) #for every training example we need to ceate a sequence input that is a 100 chars long and a sequence output thats a 100 chars long which means we need 101 chars that we use for every training example

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #converts the entire string dataset into chars. it will be a stream of chars. it will contain 1.1mil characters

sequences = char_dataset.batch(seq_length+1, drop_remainder=True) #batchings of desired length. drop remainder means when we have extra characters drop them

    #dividing the data in sequences/batches into input data and what output should be
def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry. every sequence will have this operation applied to it

    #TRAINING BATCHES
BATCH_SIZE = 64 #64 training examples(entries) that are sequences of length 100 into the model
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256 #how big we want every single vector to represent our words in the embedding layer
RNN_UNITS = 1024 

BUFFER_SIZE = 10000
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, so it doesn't attempt to shuffle the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements)

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)






#BUILDING THE MODEL
#func below will take 64 training examples(batch size) and return to us 64 outputs. later on we will rebuild the model using the same parameters weve saved and trained for the model but change the batch size to 1 to get one prediction for one input sequence
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]), #none means we dont know how long each sequence will be in each batch when we use the model to make a prediction
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True, #return the intermediate stage at every step bc we want to look at what the model is seeing at the intermediate stage not just at the final stage. we want the output at every single timestep
                        stateful=True,
                        recurrent_initializer='glorot_uniform'), #what these values are gonna start at in the LSTM
    tf.keras.layers.Dense(vocab_size) #dense layer which contains the amount of vocabulary size nodes. we want the final layer to have as many nodes as the characters in the vocabulary. this way every single one of those nodes can represent a probability distribution that that character comes next. so all those nodes' value together should give us the value of one so we can look at the last layer as a predictive layer that these chars will come next
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
# model.summary()
'''Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (64, None, 256)           16640      #64=batch size, non=len of seq,256=#of vals in vec
                                                                 
 lstm (LSTM)                 (64, None, 1024)          5246976   
                                                                 
 dense (Dense)               (64, None, 65)            66625     #65=amount of nodes which is=len of vocab
                                                                 
=================================================================
Total params: 5,330,241
Trainable params: 5,330,241
Non-trainable params: 0
_________________________________________________________________'''



 #let's have a look at a sample input and the output from our untrained model. This is so we can understand what the model is giving us.
for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape

 #LOSS FUNC
def loss(labels, logits): #takes all the labels and probability distributions(logits) and will compute how diff/similar those 2 are
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)