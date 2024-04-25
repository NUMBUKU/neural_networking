# neural_networking
With this repository I am trying to create a universal library for neural network projects. The four types I want to work on are ANN, CNN, RNN and Transformer. I am making this library to do research for my school.

As for the type of dynamic lists I use, they are vectors from the standard library. I do have a linked list library, but it is bad and not practical for this use case. The data type is double, since it is more precise. To compress this type, I used the following code:
`typedef vector<double> list; // defines a list`

The basic structure is like the following: there is a file, func.c, which defines neuron activation functions and loss functions. This file uses the math.h standard library. Then another file, neuron.cpp, defines the neuron sructure and a few basic function that come with it. It uses the vector library. At last there is the main file, neural_net.cpp, which encloses the classes that define the types of neural network. I used inheritance, where ANN is the parent and CNN, RNN and Transformer are subclasses. This file uses the iostream and fstream libraries. 

# ANN
Your good old Artificial Neural Network. It is also the parent class of all the other types.
## Public variables
* `Net net` and `Collumn outn` Vectors used to store the neurons.
* `matrix midl` and `list outl` Vectors used to store neuron activations.
* `list in` Vector to store the given input of the net.
* `int ins` Variable for storing the number of inputs to the net.
* `int out` Variable for storing the index of the highest value in neural_net::outl.
* `unsigned long iteration` Stores the number of feed forward cycles the net has gone through.
* `double coef` Variable for defining the extra coefficients in the activation functions.
* `double certainty` Variable for storing the certainty of the net.
## Public methods
### neural_net::init
Initialiser for the neural net. Takes the following arguments:
* `int inputs` The number of inputs to the net (these are not neurons, just values).
* `int outputs` The number of output neurons.
* `int layers` Self explanitory, only applies to hidden nodes.
* `int rows` Again, this one is just for the hidden nodes.
* `double coefficient` Set to `0.01` by default. Stored in `neural_net::coef`
* `bool learn_coefficients` Set to `false` by default. When set to true the net learns extra coefficients in the activation functions. If true, the value of `neural_net::coef` is used as an initialiser value. Stored in `neural_net::printnet`

# CNN

# RNN

# Transformer
