# include <math.h>

// define type
# ifdef FLOAT
    typedef float type;
# else
    typedef double type;
# endif // FLOAT

// some activation functions
// sigmoids

type binstep(type in, type a, int der);

type hyptan (type in, type a, int der){ // scaling function from -1 to 1
    if (der == 2) return 0;
    type c;
    if (der == 1) c = cosh(in);
    return der ? 1/(c*c) : tanh(in);
}

type normhyptan (type in, type a, int der){ // scaling function from 0 to 1
    if (der == 2) return 0;
    type c;
    if (der == 1) c = cosh(in);
    return der ? 1/(2 * c*c) : (tanh(in)+1)/2;
}

type arctan (type in, type a, int der){
    if (der == 2) return 0;
    return der ? 1/(1+in*in) : atan(in);
}

type normarctan (type in, type a, int der){
    if (der == 2) return 0;
    return der ? 1/(2+2*in*in) : (atan(in)+1)/2;
}

type sigmoid (type in, type a, int der){ // scaling function from 0 to 1
    if (der == 2) return 0;
    type c;
    if (der == 1) c = 1+exp(in);
    return der ? (exp(in))/(c*c) : 1/(1 + exp(-1*in));
}

// linear units

type identity (type in, type a, int der){
    if (der == 2) return 0;
    return der ? 1 : in;
}

type ReLU (type in, type a, int der){
    if (der == 2) return 0;
    return der ? (in <= 0 ? 0 : 1) : (in <= 0 ? 0 : in);
}

type LeakyReLU (type in, type a, int der){
    if (der == 1) // ∂f/∂in
        return in <= 0 ? a : 1;
    
    if (der == 2) // ∂f/∂a
        return in <= 0 ? in : 0;
    
    return in <= 0 ? in * a : in;
}

type SiLU (type in, type a, int der){
    if (der == 2) return 0;
    return der ? in * sigmoid(in, a, 1) + sigmoid(in, a, 0) : in * sigmoid(in, a, 0);
}

type ExLU (type in, type a, int der){
    if (der == 1) // ∂f/∂in
        return in <= 0 ? a * exp(in) : 1;
    
    if (der == 2) // ∂f/∂a
        return in <= 0 ? exp(in) - 1 : 0;
    
    return in <= 0 ? a * (exp(in) - 1) : in;
}

type SoftPlus (type in, type a, int der){
    if (der == 2) return 0;
    return der ? sigmoid(in,a,0) : log(1+exp(in));
}

type binstep(type in, type a, int der){
    if (der == 2) return 0;
    return der ? 0 : ReLU(in, a, 1);
}


enum act_func{ // classification for the activation functions
    BINSTEP,SIGMOID,TANH,NTANH,ARCTAN,NARCTAN,SOFTMAX, // sigmoids
    IDENTITY,RELU,LEAKYRELU,SILU,ELU,SOFTPLUS // linear units
};

type (*func(act_func f))(type, type, int){
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

type mean_squared (type y, type ypred, int der){
    type dif = y - ypred;
    return der ? 2 * dif : dif * dif;
}

type cross_entropy (type y, type ypred, int der){
    return der ? -1*ypred/y : -1*ypred*log(y);
}

type MAPE (type y, type ypred, int der){
    type dif = y - ypred;
    return der ? 100*(dif > 0 ? 1 : -1)/ypred : 100*abs(dif)/ypred;
}


enum loss_func{ // classification for the loss functions
    MEAN_SQUARED,CROSS_ENTROPY,MAPD
};

type (*lfunc(loss_func function))(type, type, int){
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