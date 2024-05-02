# include <iostream>

# include "neural_net.cpp"

using std::cout, std::ostream;

// Template to print a vector using cout. Credit: geeksforgeeks.org
template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

list input0 {0}, input1 {.5}, input2 {1},
wanted0 {1,0,0}, wanted1 {0,1,0}, wanted2 {0,0,1};

double lr (int epoch){
    return 10/(1+.01*epoch);
}

list in (int iteration){
    if (iteration == 2) return input2;
    return iteration == 0 ? input0 : input1;
}

list wa (int iteration){
    if (iteration == 2) return wanted2;
    return iteration == 0 ? wanted0 : wanted1;
}

int main(){
    neural_net net;
    net.init(1,3,3,3,LRELU,NTANH,.01,false);

    // act_func function [] = {ELU,RELU,SILU,LRELU};
    // net.modify(1, 3, 4, {4,2,1,4}, function, false, 0.01);

    double cost = 0;
    for (int epoch = 0; epoch < 100; epoch++) for (int iteration = 0; iteration < 3; iteration++){
        net.calc_out(in(iteration));
        net.improve(wa(iteration), lr(epoch), 1);
        cost += net.calc_loss(wa(iteration));
        if (iteration == 2){
            cout << "cost:" << cost/3 << "\n";
            cost = 0;
        }
    }

    list a{0};
    cout << net.calc_out(a) << " " << net.certainty << "\n\n";
    cout << net.outl << "\n\n";
    a = {.5};
    cout << net.calc_out(a) << " " << net.certainty << "\n\n";
    cout << net.outl << "\n\n";
    a = {1};
    cout << net.calc_out(a) << " " << net.certainty << "\n\n";
    cout << net.outl << "\n\n";

    return 0;
}