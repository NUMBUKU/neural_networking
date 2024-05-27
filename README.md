# neural_networking
With this repository I am trying to create a universal and open-source library for neural network projects, something like the already existing tensorflow and pytorch libraries, just much worse. The four types of NN I want to work on are ANN, CNN, RNN and LSTM. I am making this library to do research for my school.

As for the type of dynamic lists I use, they are vectors from the standard library. I do have a linked list library, but it is bad and not practical for this use case. The data type is user defined, but set to `double` by default. To compress this type, I used the following code:
```C++
typedef std::vector<type> list; // defines a list
```

To make the type user defined I used the following code:
```C++
# ifdef FLOAT
    typedef float type;
# else
    typedef double type;
# endif // FLOAT
```
So set the type to float, define FLOAT before the include line, like this:
```C++
# define FLOAT
# include "neural_net.hpp"
```
Take in mind that `double` has double the precision of `float` type, but `float` (size: 4B) requires half the memory space `double` (size: 8B) does. It is a tradeoff. `double` is the default due to personal preference.

The basic structure is like the following: there is a file, func.c, which defines neuron activation functions and loss functions. This file uses the math.h standard library. Then another file, neuron.cpp, defines the neuron sruct and a few basic functions that come with it. It uses the vector library. At last there is the main file, neural_net.cpp, which encloses the classes that define the types of neural network. I used inheritance, where ANN is the parent and CNN, RNN and LSTM are subclasses. This file uses the iostream and fstream libraries. 

# functions

## flatten
`flatten` is used


# classes

## act_func
This is an enum class. It defines a few activation functions. They are named appropiately:
```C++
enum act_func{ // classification for the activation functions
    BINSTEP,SIGMOID,TANH,NTANH,ARCTAN,NARCTAN,SOFTMAX, // sigmoids
    IDENTITY,RELU,LEAKYRELU,SILU,ELU,SOFTPLUS // linear units
};
```
BINSTEP is binarystep.
The N before some of the sigmoids means that they are normalised between zero and one.

## loss_func
This is an enum class. It defines a few loss functions. They are named appropiately:
```C++
enum loss_func{ // classification for the loss functions
    MEAN_SQUARED,CROSS_ENTROPY,MAPD
};
```
MAPD is Mean Absolute Percentage Deviation/Error

## ANN
Your good old Artificial Neural Network. It is also the parent class of all the other types. Its class is called: `neural_networking::neural_network::neural_network`. An ANN called net would be made by the following code: `neural_network net;`. Then, to add a layer to the net, `neural_network::add_dense_layer` should be run. This method is further explained later.
### Public variables
* `Net net` Vector used to store the neurons.
* `matrix actlist` Vector used to store neuron activations.
* `list in` Vector to store the given input of the net.
* `list outlist` Vector to store the activation of the neurons in the output layer.
* `type certainty` Variable for storing the certainty of the net.
* `int out_index` Set to zero by default. Variable for storing the index of the highest value in neural_network::outlist.
* `int lastlayer` Set to negative one by default. Variable for storing the index of the last layer in neural_network::net and neural_network::actlist.
* `unsigned long long iteration` Set to zero by default. Stores the number of feed forward cycles the net has gone through.
* `type coef` Variable for defining the extra coefficients in the activation functions.
* `bool learn_coefficients` Boolean for storing if the user wants the net to learn the coefficients of the activation functions by itself.
* `bool initialised` Set to false by default. Boolean for storing if the net was initialised.

### Public methods
#### neural_network::add_input
Adds the input neurons to the net.
Takes the following arguments:
* `int input_count` Amount of input neurons that should be added.

Can throw the following errors:
* `std::runtime_error("input_count should be greater than zero.");`
* `std::runtime_error("Input was already defined.");`

#### neural_network::add_dense_layer
Adds a dense neuron layer to the network. Also sets `neural_network::initialised` to true.
Takes the following arguments:
* `int neuron_count` The amount of neurons in the added layer.
* `act_func activation_function` The activation function used in the added layer. See [act_func](#act_func).
* `type coefficient` Set to 0.01 by default. Sets the coefficient used in some activation functions for this layer.

Can throw the following errors:
* `std::runtime_error("neural_network::input should have been run before adding dense layers.")`

#### neural_network::export_net


#### neural_network::evaluate
Feeds the given input through the network to predict an output.
Takes the following arguments:
* `list input` The given input to the net.

Can throw the following errors:
* `std::runtime_error("Please run neural_network::add_dense_layer to initialise the net.");`
* `std::runtime_error("Input list should be the same size as the amount of input neurons.");`

#### neural_network::loss
Calculates the loss for a single feed-forward cycle. Only works when neural_network::evaluate has already been run.
Takes the following arguments:
* `list wanted` The list of output that is the target of the net, given the previously evaluated input.
* `loss_func function` Set to MEAN_SQUARED by default. It sets the loss function by which the net should be evaluated. See [loss_func](#loss_func).

Can throw the following errors:
* `std::runtime_error("Please run neural_network::add_dense_layer to initialise the net.");`
* `std::runtime_error("Wanted list should be the same size as the number of outputs");`
* `std::runtime_error("This method only works if neural_network::evaluate has already been run.");`

#### neural_network::fit
This method is used for curve-fitting the network.
Takes the following arguments:
* `list wanted` The list of output that is the target of the net, given the previously evaluated input.
* `type learning_rate` Sets the learning rate of this iteration.
* `int batch_size` Set to one by default. It sets the size of the batches.
* `loss_func function` Set to MEAN_SQUARED by default. It sets the loss function by which the net should be evaluated. See [loss_func](#loss_func).

## CNN
### Public variables
### Public methods


## RNN

## LSTM