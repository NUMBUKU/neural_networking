# include <vector>

# include "func.c"

using std::vector;

typedef vector<double> list; // defines a list
typedef vector<list> matrix; // defines a twodimensional list
typedef struct neur_t {
    list wgt;
    double bias = 0;
    double coef = .01;
    double act = 0;
    int out = 0;
    double (*func)(double, double, int);
} neuron; // defines a neuron
typedef vector<neuron> Collumn; // defines a list of neurons
typedef vector<Collumn> Net; // defines a multidimensional list of neurons

double calc_z (list act, neuron n){ // calculates what the output of one neuron should be without scaling
    double a = 0;
    for (int i = 0; i < act.size(); i++){
        a += act[i] * n.wgt[i];
    }

    return a + n.bias;
}

double calc_act (list act, neuron n, double a){ // calculates what the output of one neuron should be
    return n.func(calc_z(act, n), a, 0);
}

double singleloss (double wanted, double given){ // calculates how bad the machine performes   
    double c = given - wanted;  
    return c*c;
}

list calc_impact (list act, neuron n, double wanted, double a){ // returns a list to indicate which variable has the most impact on the cost and how much it should change
    int size = act.size();
    list return_list;

    double dcdz = 2 * (calc_act(act, n, a) - wanted) * n.func(calc_z(act, n), a, 1);

    return_list.push_back(dcdz); //calculating the impact of the bias

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * act[i]); //calculating the impact of the weights

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * n.wgt[i]); //calculating the impact of the previous activations

    return_list.shrink_to_fit();

    return return_list;
}

list calc_impact (list act, neuron n, double a){ // returns a list to indicate which variable has the most impact on the cost and how much it should change
    int size = act.size();
    list return_list;

    double dcdz = n.func(calc_z(act, n), a, 1);

    return_list.push_back(dcdz); //calculating the impact of the bias

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * act[i]); //calculating the impact of the weights

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * n.wgt[i]); //calculating the impact of the previous activations

    return_list.shrink_to_fit();

    return return_list;
}

double (*func(act_func f))(double, double, int){
    switch (f){
        case 0:
            return &hyptan;
        case 1:
            return &normhyptan;
        case 2:
            return &arctan;
        case 3:
            return &normarctan;
        case 4:
            return &sigmoid;
        case 5:
            return &ReLU;
        case 6:
            return &LeakyReLU;
        case 7:
            return &SiLU;
        case 8:
            return &ExLU;
        case 9:
            return &SoftPlus;
        case 10:
            return &binstep;
        case 11:
            return &identity;
        default: 
            return &identity;
    }
}