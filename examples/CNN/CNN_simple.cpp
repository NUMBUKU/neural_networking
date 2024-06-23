/* 
 * This is a simple example of the Convolutional Neural Network.
 * 
 * 
 * This file trains a network to classify two basic emotions of some simple faces.
*/

// # define FLOAT // uncomment if the net should use float data type in stead of double data type
# include "..\..\neural_net.cpp"

using std::cout, std::ostream, std::endl;

const double learning_rate = 1; // learning rate althroughout the epochs

const int samples = 2, // number of training samples
    epochs = 10;       // amount of epochs

// Template to print a vector using cout. Credit: https://www.geeksforgeeks.org/different-ways-to-print-elements-of-vector/
template <typename S>
ostream& operator<< (ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

void print_output (vector<matrix> input, CNN * net){
    net->eval(input);
    cout << "index: " << net->eval(input) << " certainty:" << net->certainty << "\n"; // used when there are multiple outputs, it prints the nets certainty and and output index
    cout << "output:" << net->outlist << "\n"                                         // prints the output
    << endl; // flush
}

vector<vector<matrix>> inputlists = { // training input data
    {{ // happy face
        {0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}
    }},
    {{ // sad face
        {0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1}
    }}
};
vector<list> wantedlists = {{1,0},{0,1}}; // training output data

int main (){
    CNN net;
        net.add_input(5, 4, 1);                         // 5 by 4 input with one channel
        net.add_convolutional_layer(1, true, 3, 3, 1);  // convolution with one feature of size 3 by 3, a stride of 1 and no padding
        net.add_pooling_layer(true, 2, 2, 2);           // maxpooling with poolwindow of size 2 by 2 and a stride of 2
        net.add_dense_layer(2, SOFTMAX, 1);             // dense layer with 2 neurons and softmax activation with coefficient equal to 1
    
    double cost = 0; // initialise cost
    for (int epoch = 0; epoch < epochs; epoch++){ // an epoch is when the entire training set has been propagated forwards and backwards
        for (int iteration = 0; iteration < samples; iteration++){
            net.eval(inputlists[iteration]);                            // before fitting CNN::eval should be run
            net.fit(wantedlists[iteration], learning_rate, samples);    // fitting

            cost += net.loss(wantedlists[iteration]);                   // averaging loss to calculate cost
        }
        cout << "cost:" << cost/samples << "\n";                        // print the cost
        cost = 0;
    }

    cout << endl; // flush

    // test the net
    vector<matrix> test {{ // happy face
        {0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}
    }};
    print_output(test, &net);

    test = {{ // sad face
        {0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1}
    }};
    print_output(test, &net);

    test = {{ // neutral face
        {0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    }};
    print_output(test, &net);

    return 0;
}