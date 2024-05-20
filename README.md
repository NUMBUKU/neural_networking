# neural_networking
With this repository I am trying to create a universal and open-source library for neural network projects, something like the already existing tensorflow and pytorch libraries, just much worse. The four types of NN I want to work on are ANN, CNN, RNN and LSTM. I am making this library to do research for my school.

As for the type of dynamic lists I use, they are vectors from the standard library. I do have a linked list library, but it is bad and not practical for this use case. The data type is double, since it is more precice. To compress this type, I used the following code:
`typedef vector<double> list; // defines a list`

The basic structure is like the following: there is a file, func.c, which defines neuron activation functions and loss functions. This file uses the math.h standard library. Then another file, neuron.cpp, defines the neuron sructure and a few basic function that come with it. It uses the vector library. At last there is the main file, neural_net.cpp, which encloses the classes that define the types of neural network. I used inheritance, where ANN is the parent and CNN, RNN and LSTM are subclasses. This file uses the iostream and fstream libraries. 

# ANN
Your good old Artificial Neural Network. It is also the parent class of all the other types. Its class is called: `neural_network`. An ANN called net would be made by the following code: `neural_network net;`. Then, to initialise the net, `neural_network::init()` should be run. This method is further explained later.
## Public variables
* `Net net` Vector used to store the neurons.
* `matrix actlist` Vector used to store neuron activations.
* `list in` Vector to store the given input of the net.
* `list outlist` Vector to store the activation of the neurons in the output layer.
* `double certainty` Variable for storing the certainty of the net.
* `int out_index` Set to zero by default. Variable for storing the index of the highest value in neural_network::outlist.
* `int lastlayer` Set to negative one by default. Variable for storing the index of the last layer in neural_network::net and neural_network::actlist.
* `unsigned long iteration` Set to zero by default. Stores the number of feed forward cycles the net has gone through.
* `double coef` Variable for defining the extra coefficients in the activation functions.
* `bool learn_coefficients` Boolean for storing if the user wants the net to learn the coefficients of the activation functions by itself.
* `bool initialised` Set to false by default. Boolean for storing if the net was initialised.
## Public methods
### neural_network::init
Initialiser for the neural net. When run, it sets `neural_network::initialised` to true. Takes the following arguments:
* `int inputs` The number of inputs to the net (these are not neurons, just values).
* `int outputs` The number of output neurons.
* `int layers` Self explanitory, only applies to hidden nodes.
* `int rows` Again, this one is just for the hidden nodes.
* `double coefficient` Set to `0.01` by default. Stored in `neural_network::coef`.
* `bool learn_coefficients` Set to `false` by default. When set to true the net learns extra coefficients in the activation functions. If true, the value of `neural_network::coef` is used as an initialiser value. Stored in `neural_network::learn_coef`.
### neural_network::modify
### neural_network::print
### neural_network::calc_out
### neural_network::calc_cost
### neural_network::improve

# CNN
## Public variables
## Public methods
### CNN::initialise
### CNN::modify
### CNN::print
### CNN::calc_out
### CNN::improve

# RNN

# LSTM
