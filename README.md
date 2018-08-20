# character-level-yeezy
Implemented just to practice concepts of character level modelling.
The model is a simple LSTM cell that takes in 'INPUT_SIZE' number of characters as input, with a softmax to output the next character, in a form that can be argmax-ed to one-hot encoding of the vocabulary learnt.

dependencies: numpy, pickle, keras v2.1.2 w/ tensorflow-gpu 1.4.1
