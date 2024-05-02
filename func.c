# include <math.h>

// some activation functions
// sigmoids

double hyptan (double in, double a, int der){ // scaling function from -1 to 1
    double c;
    if (der == 1) c = cosh(in);
    return der == 1 ? 1/(c*c) : tanh(in);
}

double normhyptan (double in, double a, int der){ // scaling function from 0 to 1
    double c;
    if (der == 1) c = cosh(in);
    return der == 1 ? 1/(2 * c*c) : (tanh(in)+1)/2;
}

double arctan (double in, double a, int der){
    return der == 1 ? 1/(1+in*in) : atan(in);
}

double normarctan (double in, double a, int der){
    return der == 1 ? 1/(2+2*in*in) : (atan(in)+1)/2;
}

double sigmoid (double in, double a, int der){ // scaling function from 0 to 1
    double c;
    if (der == 1) c = 1+exp(in);
    return der == 1 ? (exp(in))/(c*c) : 1/(1 + exp(-1*in));
}

// linear units

double identity (double in, double a, int der){
    return der == 1 ? 1 : in;
}

double ReLU (double in, double a, int der){
    return der == 1 ? (in <= 0 ? 0 : 1) : (in <= 0 ? 0 : in);
}

double LeakyReLU (double in, double a, int der){
    if (der == 1) // ∂f/∂in
        return in <= 0 ? a : 1;
    
    if (der == 2) // ∂f/∂a
        return in <= 0 ? in : 0;
    
    return in <= 0 ? in * a : in;
}

double binstep(double in, double a, int der){
    return der == 1 ? 0 : ReLU(in, a, 1);
}

double SiLU (double in, double a, int der){
    return der == 1 ? in * sigmoid(in, a, 1) + sigmoid(in, a, 0) : in * sigmoid(in, a, 0);
}

double ExLU (double in, double a, int der){
    if (der == 1) // ∂f/∂in
        return in <= 0 ? a * exp(in) : 1;
    
    if (der == 2) // ∂f/∂a
        return in <= 0 ? exp(in) - 1 : 0;
    
    return in <= 0 ? a * (exp(in) - 1) : in;
}

double SoftPlus (double in, double a, int der){
    return der == 1 ? sigmoid(in,a,0) : log(1+exp(in));
}

enum act_func{ // classification for the activation functions
    TANH,NTANH,ARCTAN,NARCTAN,SIGMOID,RELU,LRELU,SILU,ELU,SP,BINSTEP,ID
};

// some loss functions

double mean_squared (double y, double ypred, int der, double C = 1){
    double dif = y - ypred;
    return der == 1 ? 2 * C * (y - ypred) : C * dif * dif;
}