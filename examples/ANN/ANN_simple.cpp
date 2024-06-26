/* 
 * This is a simple example of the Artificial Neural Network.
 * 
 * 
 * This file trains a network to return the negative of an input number.
*/

// # define FLOAT // uncomment if the net should use float data type in stead of double data type
# include "..\..\neural_net.cpp"

using std::cout, std::endl,
    std::ostream,
    std::vector;

const double learning_rate = 1; // learning rate althroughout the epochs

const int samples = 6, // number of training samples
    epochs = 10;       // amount of epochs

// Template to print a vector using cout. Credit: https://www.geeksforgeeks.org/different-ways-to-print-elements-of-vector/
// template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
//     for (auto element : vector) os << element << " ";
//     return os;
// }

void print_output (list input, ANN * net){
    net->eval(input);
    // cout << "index: " << net->out_index << " certainty:" << net->certainty << "\n"; // used when there are multiple outputs, it prints the nets certainty and and output index
    cout << "output:" << net->outlist << "\n"                                          // prints the output
    << endl; // flush
}

vector<list> inputlists = {{-2}, {1}, {-1}, {.5}, {-.5}, {2}}, // training input data
wantedlists {{2}, {-1}, {1}, {-.5}, {.5}, {-2}};               // training output data

int main (){
    ANN net;
        net.add_input(1);                    // input of size 1
        
        net.add_dense_layer(1, IDENTITY);    // dense layer with one neuron and identity activation (the coefficient isn't used in identity activation)

    double cost = 0; // initialise cost
    for (int epoch = 0; epoch < epochs; epoch++){ // an epoch is when the entire training set has been propagated forwards and backwards
        for (int iteration = 0; iteration < samples; iteration++){
            net.eval(inputlists[iteration]);                            // before fitting ANN::eval should be run
            net.fit(wantedlists[iteration], learning_rate, samples);    // fitting
            cout << "exitfit";

            cost += net.loss(wantedlists[iteration]);                   // averaging loss to calculate cost
        }
        cout << "cost:" << cost/samples << "\n";                        // print the cost
        cost = 0;
    }

    cout << endl; // flush

    // test the net
    list test {-.4};
    print_output(test, &net);

    test = {.03};
    print_output(test, &net);

    test = {600};
    print_output(test, &net);

    return 0;
}