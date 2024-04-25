# include <iostream>

# include "neural_net.cpp"

using std::cout, std::ostream;

// Template to print a vector using cout. Credit: geeksforgeeks.org
template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

list input0 {0,0,0,0}, input1 {.5,.5,.5,.5}, input2 {0,0,0,0},
wanted0 {1,0,0}, wanted1 {0,1,0}, wanted2 {0,0,1};

double lr (int epoch){
    return .1/(1+epoch);
}

list in (int iteration){
    if (iteration == 3) return input2;
    return iteration == 0 ? input0 : input1;
}

list wa (int iteration){
    if (iteration == 3) return wanted2;
    return iteration == 0 ? wanted0 : wanted1;
}

int main(){
    neural_net net;
    net.init(4, 3, 4, 2, false, 0.01);
    // act_func function [] = {ELU,RELU,SILU,LRELU};
    // net.modify(4,{4,2,1,4},4,3,function);

    for (int i = 0; i < 100; i++){
        net.calc_out(in(i % 3));
        net.improve(wa(i % 3), lr(i));
        //cout << "cost:" << net.calc_cost(wa(i % 2)) << "\n";
    }
    list a{.5,.5,.5,.5};
    cout << net.calc_out(a) << " " << net.certainty << "\n\n";
    cout << net.outl;

    return 0;
}