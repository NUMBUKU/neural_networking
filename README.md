# neural_networking
With this repository I am trying to create a universal and open-source library for neural network projects, something like the already existing tensorflow and pytorch libraries, just much worse. The four types of NN I want to work on are ANN, CNN, RNN and LSTM. I am making this library to do research for my school. This is a work in progress, so not everything is finished or functional.

As for the type of dynamic lists I use, they are vectors from the standard library. I do have a linked list library, but it is bad and not practical for this use case.

The basic structure is like the following: there is a file, func.c, which defines neuron activation functions and loss functions. This file uses the math.h standard library. Then another file, neuron.cpp, defines the neuron sruct and a few basic functions that come with it. It uses the vector library. At last there is the main file, neural_net.cpp, which encloses the classes that define the types of neural network. I used inheritance, where ANN is the parent and CNN, RNN and LSTM are subclasses. This file uses the iostream and fstream libraries. 

# typedefs

### type
I wanted to make the type user defined so I used the following code:
```C++
# ifdef FLOAT
    typedef float type; // use float
# else
    typedef double type; // use double
# endif // FLOAT
```
`double` is default. To set the type to `float`, define FLOAT before the include line, like this:
```C++
# define FLOAT // define FLOAT to use float type
# include "neural_net.cpp" // include line
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

## neural_networking::act_func
This is an enum class. It defines a few activation functions. They are named appropiately:
```C++
enum act_func{ // classification for the activation functions
    BINSTEP,SIGMOID,TANH,NTANH,ARCTAN,NARCTAN,SOFTMAX, // sigmoids
    IDENTITY,RELU,LEAKYRELU,SILU,ELU,GELU,SOFTPLUS // linear units
};
```
BINSTEP is binarystep and SILU is Sigmoid weighted Linear Unit, also called swish.
The N before some of the sigmoids means that they are normalised between zero and one.

Some functions use coëfficients/paramaters, they are: softmax, SiLU, LeakyReLU and ELU. The parameters are placed like this (β is the parameter):

$$softmax: σ(\vec{v})_i = {e^{βv_i} \over {\sum_j{e^{βv_j}}}}$$

$$
LeakyReLU(x) =
  \begin{cases}
    x & x≥0 \\
    β·x & x<0 \\
  \end{cases}
$$

$$SiLU(x) = {x \over {1+e^{-βx}}}$$

$$
ELU(x) =
  \begin{cases}
    x & x≥0 \\
    β·(e^{x}-1) & x<0 \\
  \end{cases}
$$

GELU is approximated as $x·σ(1.702·x)$ (σ is the sigmoid) as suggested in [this paper](https://arxiv.org/pdf/1606.08415v5).
> We can approximate the GELU with xσ(1.702x), if greater feedforward speed is worth the cost of exactness.

## neural_networking::loss_func
This is an enum class. It defines a few loss functions. They are named appropiately:
```C++
enum loss_func{ // classification for the loss functions
    MEAN_SQUARED,NMEANSQUARED,CROSS_ENTROPY,MAPD
};
```
MAPD is Mean Absolute Percentage Deviation/Error and NMEANSQUARED is mean squared loss multiplied by a half to 'normalise' it by multiplying it by 0.5 to lose the constant factor in the derivative. cross entropy uses ln, so to stop $ln(0)$ cases from happening I set the minimum log value to 30,000.

There are multiple ways to define these functions, but these are the ones I used:

mean squared loss:

$$\mathcal{L} = (y - y_{pred})^2$$

normalised mean square loss:

$$\mathcal{L} = {1 \over 2} (y - y_{pred})^2$$

cross entropy:

$$\mathcal{L} = -y_{pred}·ln(y) - (1 - y_{pred})·ln(1 - y)$$

MAPD:

$$\mathcal{L} = 100\left\lvert{{y_{pred}-y \over y_{pred}}}\right\rvert$$


Here, y<sub>pred</sub> is the expected output and y is the output of the net.

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
* `list outlist` Vector for easier acces to the activation of the neurons in the output layer.
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
* `act_func activation_function` The activation function used in the added layer. See [act_func](#neural_networkingact_func).
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
* `loss_func function` Set to MEAN_SQUARED by default. It sets the loss function by which the net should be evald. See [loss_func](#neural_networkingloss_func).

Can throw the following errors:
* `std::runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");`
* `std::runtime_error("Wanted list should be the same size as the number of outputs");`
* `std::runtime_error("This method only works if ANN::eval has already been run.");`

Returns the calculated loss.

Loss is not the same as cost: loss is the error for one single training sample. Cost however, is the average of all the losses of all the samples in one trining set. For a training set with N samples, cost is calculated like this:

$$\mathcal{C} = {1 \over N}\sum_{k=1}^{N} \mathcal{L}_k$$

Where $\mathcal{L}_k$ is the loss for training sample k.

---
#### ANN::fit
This method is used for curve-fitting the network.
Takes the following arguments:
* `list wanted` The list of output that is the target of the net, given the previously evald input.
* `type learning_rate` Sets the learning rate of this iteration.
* `int batch_size` Set to one by default. It sets the size of the batches.
* `loss_func function` Set to MEAN_SQUARED by default. It sets the loss function by which the net should be evald. See [loss_func](#neural_networkingloss_func).

Can throw the following errors:
* `std::runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");`
* `std::runtime_error("Wanted list should be the same size as the number of outputs")`
* `std::runtime_error("This method only works if ANN::eval has already been run.")`

No return value.

## neural_networking::CNN
A Convolutional Neural Network, made for image classification but has wide applications. This is a subclass inherited from the ANN class. A CNN called net with a ten by ten input, one convolution layer, one maxpooling layer and one neuron with relu activation would be defined like this:
```C++
CNN net;                                // create the CNN
net.input(10, 10);                      // add the input
net.add_convolutional_layer(1, false);  // add a convolutional layer
net.add_pooling_layer(true);            // add a maxpooling layer
net.add_dense_layer(1, RELU);           // add a neuron layer
```
### Public variables
* `vector<vector<matrix>> K` Tensor for storing the kernells/features of the convolutional layers.
* `vector<vector<matrix>> channels` Tensor for storing the output of every conlayer.
* `matrix biasses` vector of vectors for storing the biasses used in convolutional layers.

### Public methods
#### CNN::add_input
You would never guess what this function does.
Takes the following arguments:
* `int xinput` The horizontal size of the input matrix.
* `int yinput` The vertical size of the input matrix.
* `int channelcount` Set to one by default. Used for example when an image has Red, Green and Blue channels.

Can throw the following errors:
* `std::runtime_error("All parameters of CNN::input should be greater than zero.")`

No return value.

---
#### CNN::add_convolutional_layer
This function adds a convolutional layer to the network. Sometimes I use convolutional layer as a term for a layer which performs, well, convolution. But sometimes I use it as a term for a layer inherent to a CNN (so a pooling- or convolutional layer).
Takes the following arguments:
* `int features` The amount of features that should be used in this layer.
* `bool paddingvalid` When false padding is set to 'same', which means padding will be used to make the convolutional layer output the same size as the input. Otherwise no padding will be used in the 'valid' mode.
* `int xfeature` Set to 3 by default. Horizontal size of the feature.
* `int yfeature` Set to 3 by default. Vertical size of the feature.
* `int stridefeature` Set to 1 by default. 'Stride' of the feature (how much it moves) both in horizontal and vertical direction.

Can throw the following errors:
* `std::runtime_error("No convolutional layers should be added after adding a fully connected layer.")`
* `std::runtime_error("CNN::add_input should have been run before adding dense layers.")`

No return value.

---
#### CNN::add_pooling_layer
This function does the same as `CNN::add_convolutional_layer`, except it adds a pooling layer instead of a convolutional layer.
Takes the following arguments:
* `bool maxpooling` When set to true this layer will perform 'maxpooling' and otherwise it will do 'averagepooling'.
* `int xpoolwindow` Set to 2 by default. Horizontal size of the poolwindow.
* `int ypoolwindow` Set to 2 by default. Vertical size of the poolwindow.
* `int stridepoolwindow` Set to 2 by default. 'Stride' of the poolwindow both in horizontal and vertical direction.

Can throw the following errors:
* `std::runtime_error("No pooling layers should be added after adding a fully connected layer.")`

No return value.

---
#### CNN::add_dense_layer
Very similar to [ANN::add_dense_layer](#annadd_dense_layer), only it can throw another error:
* `std::runtime_error("Please add convolutional or pooling layers first.")`

---
#### CNN::eval
Similar to [ANN::eval](#anneval). It has two overloads, one with a matrix as input and another with a vactro of matrices as input (if the user wants to use multiple channels)

They can both throw the following error:
* `std::runtime_error("Please run CNN::add_convolutional_layer or CNN::add_pooling_layer and then CNN::add_dense_layer to initialise the CNN.")`

---
#### CNN::loss
Very similar to [ANN::loss](#annloss).

Can throw the following errors:
* `std::runtime_error("Please run CNN::add_convolutional_layer or CNN::add_pooling_layer and then CNN::add_dense_layer to initialise the CNN.")`
* `std::runtime_error("This method only works if CNN::eval has already been run.")`
* `std::runtime_error("Wanted list should be the same size as the number of outputs")`

---
#### CNN::fit
Very Similar to [ANN::fit](#annfit).

Can throw the following errors:
* `std::runtime_error("Please run CNN::add_convolutional_layer or CNN::add_pooling_layer and then CNN::add_dense_layer to initialise the CNN.")`
* `std::runtime_error("Wanted list should be the same size as the number of outputs")`
* `std::runtime_error("This method only works if ANN::eval has already been run.")`

## neural_networking::RNN

## neural_networking::LSTM
