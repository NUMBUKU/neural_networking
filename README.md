# neural_networking
With this repository I am trying to create a universal and open-source library for neural network projects, something like the already existing tensorflow and pytorch libraries, just much worse. The four types of NN I want to work on are ANN, CNN, RNN and LSTM. I am making this library to do research for my school.

As for the type of dynamic lists I use, they are vectors from the standard library. I do have a linked list library, but it is bad and not practical for this use case.

The basic structure is like the following: there is a file, func.c, which defines neuron activation functions and loss functions. This file uses the math.h standard library. Then another file, neuron.cpp, defines the neuron sruct and a few basic functions that come with it. It uses the vector library. At last there is the main file, neural_net.cpp, which encloses the classes that define the types of neural network. I used inheritance, where ANN is the parent and CNN, RNN and LSTM are subclasses. This file uses the iostream and fstream libraries. 

# typedefs

### type
I wanted to make the type user defined so I used the following code:
```C++
# ifdef FLOAT
    typedef float type;
# else
    typedef double type;
# endif // FLOAT
```
`double` is default. To set the type to `float`, define FLOAT before the include line, like this:
```C++
# define FLOAT
# include "neural_net.cpp"
```
Take in mind that `double` has double the precision of `float` type, but `float` (size: 4B) requires half the memory space `double` (size: 8B) does. It is a tradeoff. `double` is the default due to personal preference.

---
### list and matrix
`list` is a vector of `type` defined like this:
```C++
typedef std::vector<type> list; // defines a list
```
`matrix` is a vector of lists, defined like this:
```C++
typedef std::vector<list> matrix; // defines a twodimensional list
```
These are both defined outside the neural_networking namespace.

---
### neural_networking::neuron
`neural_networking::neuron` is a structure which entails all the variables needed to define a neuron/node of a neural net.

---
### neural_networking::Collumn and neural_networking::net
In esscence the same as list and matrix, just heat they are vectors of neurons, defined like this:
```C++
typedef std::vector<neuron> Collumn; // defines a list of neurons
typedef std::vector<Collumn> Net; // defines a twodimensional list of neurons
```

# functions

### neural_networking::flatten
`neural_networking::flatten` is used to, well, flatten an input matrix into a vector. It goes left to right and down.
Takes the following arguments:
* `matrix in` The input matrix to be flattened

Returns the vector of the flattened elements.

---
### neural_networking::softmax
`neural_networking::softmax` is the famous function used throughout data analysis, probability and such. (you should just look it up)
Takes the following arguments:
* `list in` the input vector to be normalised.
* `type a` a coefficient which, in formulas, is commonly denoted as bèta (β)

returns the normalised vector.


# classes

### neural_networking::act_func
This is an enum class. It defines a few activation functions. They are named appropiately:
```C++
enum act_func{ // classification for the activation functions
    BINSTEP,SIGMOID,TANH,NTANH,ARCTAN,NARCTAN,SOFTMAX, // sigmoids
    IDENTITY,RELU,LEAKYRELU,SILU,ELU,GELU,SOFTPLUS // linear units
};
```
BINSTEP is binarystep and SILU is Sigmoid weighted Linear Unit, also called swish.
The N before some of the sigmoids means that they are normalised between zero and one.

LeakyReLU and ELU are parametric.
GELU is approximated as x·σ(1.702·x) as suggested in [this paper](https://arxiv.org/pdf/1606.08415v5).

### neural_networking::loss_func
This is an enum class. It defines a few loss functions. They are named appropiately:
```C++
enum loss_func{ // classification for the loss functions
    MEAN_SQUARED,NMEANSQUARED,CROSS_ENTROPY,MAPD
};
```
MAPD is Mean Absolute Percentage Deviation/Error and NMEANSQUARED is mean squared loss multiplied by a half to 'normalise' it by losing the constant factor in the derivative.

## neural_networking::ANN
Your good old Artificial Neural Network. It is also the parent class of all the other types. An ANN called net with one input and one neuron would be defined like this:
```C++
ANN net;                        // create the ANN
net.add_input(1);               // add the input
net.add_dense_layer(1, RELU);   // add a neuron layer
```
### Public variables
* `Net net` Vector used to store the neurons.
* `matrix actlist` Vector used to store neuron activations.
* `list outlist` Vector to store the activation of the neurons in the output layer.
* `type certainty` Variable for storing the certainty of the net.
* `int out_index` Set to zero by default. Variable for storing the index of the highest value in ANN::outlist.
* `unsigned long long iteration` Set to zero by default. Stores the number of feed forward cycles the net has gone through.

### Public methods
#### ANN::add_input
Adds the input neurons to the net.
Takes the following arguments:
* `int input_count` Amount of input neurons that should be added.

Can throw the following errors:
* `std::runtime_error("input_count should be greater than zero.");`
* `std::runtime_error("Input was already defined.");`

No return value. `input_count` is also stored in `ANN::incount`.

---
#### ANN::add_dense_layer
Adds a dense neuron layer to the network. Also sets `ANN::initialised` to true.
Takes the following arguments:
* `int neuron_count` The amount of neurons in the added layer.
* `act_func activation_function` The activation function used in the added layer. See [act_func](#act_func).
* `type coefficient` Set to 0.01 by default. Sets the coefficient used in some activation functions for this layer.

Can throw the following errors:
* `std::runtime_error("ANN::input should have been run before adding dense layers.")`

No return value.

---
#### ANN::export_net
This function is used to print the nets parameters to export the somewhere else, in code format, So, if you wanted to export your net, you would need to have the library installed at that location as well. I am currently wrking on a way to make this more efficient and user friendly.
This function can write the data to the terminal but it can also make a new file and write data to that.
Takes the following arguments:
* `bool write_to_terminal` Set to true by default. Specifies if the user wants to write the data to the terminal.
* `bool write_to_file` Set to false by default. Specifies if the user wants to write the data to a file. 
* `const char * path` Set to "data.txt" by default. Sets the path that the file should be made at or the path of the file it should write to.

Can throw the following errors:
* `std::runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");`

No return value.

---
#### ANN::eval
Feeds the given input through the network to predict an output.
Takes the following arguments:
* `list input` The given input to the net.

Can throw the following errors:
* `std::runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");`
* `std::runtime_error("Input list should be the same size as the amount of input neurons.");`

Returns the index of the highest value in `ANN::outlist`. This is also stored in `ANN::out_index`

---
#### ANN::loss
Calculates the loss for a single feed-forward cycle. Only works when ANN::eval has already been run.
Takes the following arguments:
* `list wanted` The list of output that is the target of the net, given the previously evald input.
* `loss_func function` Set to MEAN_SQUARED by default. It sets the loss function by which the net should be evald. See [loss_func](#loss_func).

Can throw the following errors:
* `std::runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");`
* `std::runtime_error("Wanted list should be the same size as the number of outputs");`
* `std::runtime_error("This method only works if ANN::eval has already been run.");`

Returns the calculated loss.

---
#### ANN::fit
This method is used for curve-fitting the network.
Takes the following arguments:
* `list wanted` The list of output that is the target of the net, given the previously evald input.
* `type learning_rate` Sets the learning rate of this iteration.
* `int batch_size` Set to one by default. It sets the size of the batches.
* `loss_func function` Set to MEAN_SQUARED by default. It sets the loss function by which the net should be evald. See [loss_func](#loss_func).

Can throw the following errors:
* `std::runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");`

No return value.

## neural_networking::CNN
A Convolutional Neural Network, made for image classification but has wide applications. A CNN called net with a ten by ten input, one convolution layer, one maxpooling layer and one neuron with relu activation would be defined like this:
```C++
CNN net;                                // create the CNN
net.input(10, 10);                      // add the input
net.add_convolutional_layer(1, false);  // add a convolutional layer
net.add_pooling_layer(true);            // add a maxpooling layer
net.add_dense_layer(1, RELU);           // add a neuron layer
```
### Public variables
* `vector<vector<matrix>> K`
* `vector<vector<matrix>> channels`
* `vector<vector<matrix>> backprop`
* `matrix biasses`
* `matrix conlayers`
* `int xin, yin`
* `int inANN`
* `int outchan`
* `int lastconlayer`
### Public methods


## neural_networking::RNN

## neural_networking::LSTM
