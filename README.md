# neural_networking
With this repository I am trying to create a universal library for neural network projects. The four types I want to work on are ANN, CNN, RNN and Transformer. I am making this library to do research for my school.

As for the type of dynamic lists I use, they are vectors from the standard library. I do have a linked list library, but it is bad and not practical for this use case. The data type is double, since it is more precise.

The basic structure is like the following: there is a file, func.c, which defines neuron activation functions, loss functions and pooling functions (for CNN). This file uses the math.h standard library. Then another file, neuron.cpp, defines the neuron sructure and a few basic function that come with it. It uses the vector library. At last there is the main file, neural_net.cpp, which encloses the classes that define the types of neural network. I used inheritance, where ANN is the parent and CNN, RNN and Transformer are subclasses. This file uses the iostream and fstream libraries. 

# ANN
Your good old Artificial Neural Network. Does what it should do. It is also the parent class of all the other types.
## Methods:
### neural_net::init
Initialiser for the neural net. Takes the following arguments:
`int inputs`    The number of inputs to the net (these are not neurons, just values).
`int outputs`   The number of output neurons.
`int layers`    Self explanitory, only applies to hidden nodes.
`int rows`      Again, this one is just for the hidden nodes.
`bool printnet_after_death` 

# CNN

# RNN

# Transformer
