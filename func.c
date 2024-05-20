# include <math.h>

// some activation functions
// sigmoids

double binstep(double in, double a, int der);

double hyptan (double in, double a, int der){ // scaling function from -1 to 1
    double c;
    if (der == 1) c = cosh(in);
    return der ? 1/(c*c) : tanh(in);
}

double normhyptan (double in, double a, int der){ // scaling function from 0 to 1
    double c;
    if (der == 1) c = cosh(in);
    return der ? 1/(2 * c*c) : (tanh(in)+1)/2;
}

double arctan (double in, double a, int der){
    return der ? 1/(1+in*in) : atan(in);
}

double normarctan (double in, double a, int der){
    return der ? 1/(2+2*in*in) : (atan(in)+1)/2;
}

double sigmoid (double in, double a, int der){ // scaling function from 0 to 1
    double c;
    if (der == 1) c = 1+exp(in);
    return der ? (exp(in))/(c*c) : 1/(1 + exp(-1*in));
}

// linear units

double identity (double in, double a, int der){
    return der ? 1 : in;
}

double ReLU (double in, double a, int der){
    return der ? (in <= 0 ? 0 : 1) : (in <= 0 ? 0 : in);
}

double LeakyReLU (double in, double a, int der){
    if (der == 1) // ∂f/∂in
        return in <= 0 ? a : 1;
    
    if (der == 2) // ∂f/∂a
        return in <= 0 ? in : 0;
    
    return in <= 0 ? in * a : in;
}

double SiLU (double in, double a, int der){
    return der ? in * sigmoid(in, a, 1) + sigmoid(in, a, 0) : in * sigmoid(in, a, 0);
}

double ExLU (double in, double a, int der){
    if (der == 1) // ∂f/∂in
        return in <= 0 ? a * exp(in) : 1;
    
    if (der == 2) // ∂f/∂a
        return in <= 0 ? exp(in) - 1 : 0;
    
    return in <= 0 ? a * (exp(in) - 1) : in;
}

double SoftPlus (double in, double a, int der){
    return der ? sigmoid(in,a,0) : log(1+exp(in));
}

double binstep(double in, double a, int der){
    return der ? 0 : ReLU(in, a, 1);
}


enum act_func{ // classification for the activation functions
    BINSTEP,SIGMOID,TANH,NTANH,ARCTAN,NARCTAN,SOFTMAX,
    IDENTITY,RELU,LEAKYRELU,SILU,ELU,SOFTPLUS
};

double (*func(act_func f))(double, double, int){
    switch (f){
        case TANH:
            return &hyptan;
        case NTANH:
            return &normhyptan;
        case ARCTAN:
            return &arctan;
        case NARCTAN:
            return &normarctan;
        case SIGMOID:
            return &sigmoid;
        case RELU:
            return &ReLU;
        case LEAKYRELU:
            return &LeakyReLU;
        case SILU:
            return &SiLU;
        case ELU:
            return &ExLU;
        case SOFTPLUS:
            return &SoftPlus;
        case BINSTEP:
            return &binstep;
        case IDENTITY:
            return &identity;
        default: 
            return &identity;
    }
}

// some loss functions

double mean_squared (double y, double ypred, int der){
    double dif = y - ypred;
    return der ? 2 * dif : dif * dif;
}

double cross_entropy (double y, double ypred, int der){
    return der ? -1*ypred/y : -1*ypred*log(y);
}

double MAPE (double y, double ypred, int der){
    double dif = y - ypred;
    return der ? 100*(dif > 0 ? 1 : -1)/ypred : 100*abs(dif)/ypred;
}


enum loss_func{
    MEAN_SQUARED,CROSS_ENTROPY,MAPD
};

double (*lfunc(loss_func function))(double, double, int){
    switch (function){
        case MEAN_SQUARED:
            return &mean_squared;
        case CROSS_ENTROPY:
            return &cross_entropy;
        case MAPD:
            return &MAPE;
        default:
            return &mean_squared;
    }
}