import tensorflow_probability as tfp
import tensorflow as tf



# We will model a simple weather system and try to predict the temperature on each day given the following information.
# 1.Cold days are encoded by a 0 and hot days are encoded by a 1.
# 2.The first day in our sequence has an 80% chance of being cold.
# 3.A cold day has a 30% chance of being followed by a hot day.
# 4.A hot day has a 20% chance of being followed by a cold day.
# 5.On each day the temperature is normally distributed with mean and standard deviation of 0 and 5 when its a cold day. however a mean and standard deviation of 15 and 10 when its a hot day.


# CREATING DISTRIBUTION VARIABLE
tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # Refer to point 2 above. Categorical is a method of actually doing the distribution. 0.8 = 80% of being cold and 0.2 = 20% of being hot 
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above. the loc argument represents the mean (average temp for cold/hot days) and the scale is the standard devitation (by how much each side of our average value will be minused and added)

# CREATING THE MODEL
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7) #steps is how many days we want to predict for. the number of steps is how many times we wil step through this probability cycle and run the model



# RESULTS FOR EXPECTED TEMPS FOR EACH DAY
mean = model.mean() #calculates the probability
# due to the way TensorFlow works on a lower level we need to evaluate part of the graph from within a session to see the value (of this tensor)
with tf.compat.v1.Session() as sess:  
  print(mean.numpy()) # our outputs were "2.9999998, 8.4, 10.02, 10.506, 10.651799, 10.69554, 10.708661" which makes sense as we start with a high likelihood of a cold day