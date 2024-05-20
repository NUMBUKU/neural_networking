# include <vector>

# include "func.c"

using std::vector, std::copy;

typedef vector<double> list; // defines a list
typedef vector<list> matrix; // defines a twodimensional list
typedef struct neur_t {
    list wgt;
    double bias = 0;
    double coef;
    double (*func)(double, double, int);
    bool softmax = false;
    double previouschange = 0;
} neuron; // defines a neuron
typedef vector<neuron> Collumn; // defines a list of neurons
typedef vector<Collumn> Net; // defines a multidimensional list of neurons

list softmax (list in, double a){
    int l = in.size();
    list returnval (l);
    
    double sum = 1;
    for (int i = 0; i < l; i++) sum += exp(a*in[i]);

    for (int i = 0; i < l; i++){
        returnval[i] = exp(a*in[i])/sum;
    }

    return returnval;
}

list flatten (matrix in){
    int col = in.size(); int rows = in[0].size();
    list out (col * rows);

    for (int i = 0; i < col; i++) for (int j = 0; j < rows; j++)
        out.push_back(in[i][j]);
    out.shrink_to_fit();

    return out;
}

bool insmatrix (matrix l1, matrix l2, int xpa, int ypa){
    if (l2.size() != l1.size() + 2*ypa) return 1;
    int size = l1[0].size();
    for (int i = 0; i < l2.size(); i++){
        if (l2[i].size() != size + 2*xpa) return 1;
        copy(l2[i].begin(), l2[i].end(), l1[i+ypa].begin() + xpa);
    }
    return 0;
}


double calc_z (list act, neuron n){ // calculates what the output of one neuron should be without scaling
    double a = 0;
    for (int i = 0; i < act.size(); i++){
        a += act[i] * n.wgt[i];
    }

    return a + n.bias;
}

double calc_act (list act, neuron n){ // calculates what the output of one neuron should be
    return n.softmax ? calc_z(act, n) : n.func(calc_z(act, n), n.coef, 0);
}

list calc_impact (list act, neuron n, double wanted, double out, loss_func function){ // returns a list to indicate which variable has the most impact on the cost and how much it should change
    int size = act.size();
    list return_list;

    double dcdz = lfunc(function)(out, wanted, 1);
    if (n.softmax) dcdz *= out - out*out;
    else dcdz *= n.func(calc_z(act, n), n.coef, 1);

    return_list.push_back(dcdz); //calculating the impact of the bias

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * act[i]); //calculating the impact of the weights

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * n.wgt[i]); //calculating the impact of the previous activations

    return_list.shrink_to_fit();

    return return_list;
}

list calc_impact (list act, neuron n, double out){ // returns a list to indicate which variable has the most impact on the cost and how much it should change
    int size = act.size();
    list return_list;

    double dcdz;
    if (n.softmax) dcdz = out - out*out;
    else dcdz = n.func(calc_z(act, n), n.coef, 1);

    return_list.push_back(dcdz); //calculating the impact of the bias

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * act[i]); //calculating the impact of the weights

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * n.wgt[i]); //calculating the impact of the previous activations

    return_list.shrink_to_fit();

    return return_list;
}