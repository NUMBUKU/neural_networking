# include <iostream>

// # define FLOAT
# include "neural_net.cpp"

using std::cout, std::ostream;

const double learning_rate = 1;

// Template to print a vector using cout. Credit: geeksforgeeks.org
template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

int main (){
    

    return 0;
}