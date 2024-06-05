// # define FLOAT
# include "..\neural_net.cpp"

using std::cout, std::endl,
    std::ostream,
    std::vector;

const double learning_rate = 1;

// Template to print a vector using cout. Credit: geeksforgeeks.org
template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

void print_output (list input, ANN net){
    // cout << "index: " << net.evaluate(input) << " certainty:" << net.certainty << "\n"; // used when there are multiple outputs
    cout << "output:" << net.outlist << "\n" << endl;
}

vector<list> inputlists = {{-2}, {1}, {-1}, {.5}, {-.5}, {2}},
wantedlists {{2}, {-1}, {1}, {-.5}, {.5}, {-2}};

int main (){
    ANN net;
    net.add_input(1);
    net.add_dense_layer(1, IDENTITY, 0.01);

    double cost = 0;
    for (int epoch = 0; epoch < 10; epoch++) for (int iteration = 0; iteration < 6; iteration++){
        net.evaluate(inputlists[iteration]);

        net.fit(wantedlists[iteration], learning_rate, 6);

        cost += net.loss(wantedlists[iteration]);
        if (iteration == 5){
            cout << "cost:" << cost/6 << "\n";
            cost = 0;
        }
    }

    cout << endl;

    list a{-.4};
    print_output(a, net);
    a = {.03};
    print_output(a, net);
    a = {600};
    print_output(a, net);

    return 0;
}