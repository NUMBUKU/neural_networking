# include <vector>

# include "func.c"

using std::vector;

typedef vector<type> list; // defines a list
typedef vector<list> matrix; // defines a twodimensional list
typedef struct {
    list wgt;
    type bias = 0;

    // list dcdw; // das 28-32B erbij per neuron!! doe anders
    // type dcdb;

    type (*func)(type, type, int);
    type coef;
    bool softmax = false;
} neuron; // defines a neuron
typedef vector<neuron> Collumn; // defines a list of neurons
typedef vector<Collumn> Net; // defines a twodimensional list of neurons


const int CONVDETAILS = 7, // amount of details used to define a convolutional layer
    POOLDETAILS = 5; // amount of details used to define a pooling layer

typedef struct {
    bool type;
    union {
        int convolutional[CONVDETAILS];
        int pooling[POOLDETAILS];
    };
    
} layerdetails;

list softmax (list in, type a){ // softmax activation function
    int l = in.size();
    list returnval (l);
    
    type sum = 1;
    for (int i = 0; i < l; i++) sum += exp(a*in[i]);

    for (int i = 0; i < l; i++){
        returnval[i] = exp(a*in[i])/sum;
    }

    return returnval;
}

list flatten (matrix in){ // function to flatten a matrix into a list
    int col = in.size();
    list out;

    for (int i = 0; i < col; i++) for (int j = 0; j < in[i].size(); j++)
        out.push_back(in[i][j]);
    out.shrink_to_fit();

    return out;
}

bool insmatrix (matrix l1, matrix l2, int xpa, int ypa){ // function to insert a matrix into a bigger metrix, this is used to add padding
    if (l2.size() != l1.size() + 2*ypa) return 1;
    int size = l1[0].size();
    for (int i = 0; i < l2.size(); i++){
        if (l2[i].size() != size + 2*xpa) return 1;
        copy(l2[i].begin(), l2[i].end(), l1[i+ypa].begin() + xpa);
    }
    return 0;
}


type calc_z (list act, neuron * n){ // calculates what the output of one neuron should be without scaling
    type a = 0;
    for (int i = 0; i < act.size(); i++){
        a += act[i] * n->wgt[i];
    }

    return a + n->bias;
}

type calc_act (list act, neuron * n){ // calculates what the output of one neuron should be
    return n->softmax ? calc_z(act, n) : n->func(calc_z(act, n), n->coef, 0);
}

list calc_impact (list act, neuron * n, type wanted, type out, loss_func function, bool lcoef){ // returns a list to indicate which variable has the most impact on the cost and how much it should change
    int size = act.size();
    list return_list;

    type z = calc_z(act, n),
    dcdz = lfunc(function)(out, wanted, 1);

    if (lcoef) return_list.push_back(dcdz * n->func(z, n->coef, 2));

    if (n->softmax) dcdz *= out - out*out;
    else dcdz *= n->func(z, n->coef, 1);

    return_list.push_back(dcdz); //calculating the impact of the bias

    for (int i = 0; i < size; i++){
        return_list.push_back(dcdz * act[i]); //calculating the impact of the weights
        return_list.push_back(dcdz * n->wgt[i]); //calculating the impact of the previous activations
    }
    return_list.shrink_to_fit();

    return return_list;
}

list calc_impact (list act, neuron * n, type out, bool lcoef){ // returns a list to indicate which variable has the most impact on the cost and how much it should change
    int size = act.size();
    list return_list;

    type z = calc_z(act, n),
    dcdz;

    if (lcoef) return_list.push_back(n->func(z, n->coef, 2));
    
    if (n->softmax) dcdz = out - out*out;
    else dcdz = n->func(z, n->coef, 1);

    return_list.push_back(dcdz); //calculating the impact of the bias

    for (int i = 0; i < size; i++) return_list.push_back(dcdz * act[i]); //calculating the impact of the weights

    return_list.shrink_to_fit();

    return return_list;
}