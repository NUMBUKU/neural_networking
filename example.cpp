# include <iostream>

# include "neural_net.cpp"

using std::cout, std::ostream;

const type const_learning_rate = 1,
    initial_learning_rate = .2,
    decay = 1;


// Template to print a vector using cout. Credit: geeksforgeeks.org
template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

vector<list> inputlists = {{0}, {1}, {-1}, {.5}, {-.5}, {2}},
wantedlists {{0}, {-1}, {1}, {-.5}, {.5}, {-2}};


type lr (int epoch){
    return initial_learning_rate/(1+decay*epoch);
    // return const_learning_rate;
}

int main (){
    neural_network net;
    net.add_dense_layer(1, IDENTITY, 1, 0.01);

    type cost = 0;
    for (int epoch = 0; epoch < 50; epoch++) for (int iteration = 0; iteration < 6; iteration++){
        net.evaluate(inputlists[iteration]);
        net.fit(wantedlists[iteration], lr(epoch), 1);
        cost += net.calc_loss(wantedlists[iteration]);
        if (iteration == 5){
            cout << "cost:" << cost/6 << "\n";
            cost = 0;
        }
    }

    cout << net.net[0][0].wgt;

    list a{-.4};
    cout << net.evaluate(a) << " " << net.certainty << "\n\n";
    cout << net.outlist << "\n\n";
    a = {.03};
    cout << net.evaluate(a) << " " << net.certainty << "\n\n";
    cout << net.outlist << "\n\n";
    a = {600};
    cout << net.evaluate(a) << " " << net.certainty << "\n\n";
    cout << net.outlist << "\n\n";

    return 0;
}