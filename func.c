# include <math.h>

// define type
# ifdef FLOAT
    typedef float type;
# else
    typedef double type;
# endif // FLOAT

const double MIN_LOG_VAL = -30'000;

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
    switch (der){
        case 1:
            return 1/(1+in*in);
        case 2:
            return 0;
        default:
            return atan(in);
    }
}

type normarctan (type in, type a, int der){
    switch (der){
        case 1:
            return 1/(2+2*in*in);
        case 2:
            return 0;
        default:
            return (atan(in)+1)/2;
    }
}

type sigmoid (type in, type a, int der){ // scaling function from 0 to 1
    if (der == 2) return 0;
    type c;
    if (der == 1) c = 1+exp(in);
    return der ? (exp(in))/(c*c) : 1/(1 + exp(-1*in));
}

// linear units

type identity (type in, type a, int der){
    switch (der){
        case 1:
            return 1;
        case 2:
            return 0;
        default:
            return in;
    }
}

type ReLU (type in, type a, int der){
    switch (der){
        case 1:
            return in < 0 ? 0 : 1;
        case 2:
            return 0;
        default:
            return in < 0 ? 0 : in;
    }
}

type LeakyReLU (type in, type a, int der){
    switch (der){
        case 1:
            return in < 0 ? a : 1;
        case 2:
            return in < 0 ? in : 0;
        default:
            return in < 0 ? in * a : in;
    }
}

type SiLU (type in, type a, int der){
    switch (der){
        case 1:
            return in * a * sigmoid(a*in, a, 1) + sigmoid(a*in, a, 0);
        case 2:
            return in * in * sigmoid (a*in, a, 1);
        default:
            return in * sigmoid(a*in, a, 0);
    }
}

type ExLU (type in, type a, int der){
    switch (der){
        case 1:
            return in < 0 ? a * exp(in) : 1;
        case 2:
            return in < 0 ? exp(in) - 1 : 0;
        default:
            return in < 0 ? a * (exp(in) - 1) : in;
    }
}

type GeLU (type in, type a, int der){
    return SiLU(in, 1.702, der); // an aproximation of Gaussian Error Linear Unit
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
    IDENTITY,RELU,LEAKYRELU,SILU,ELU,GELU,SOFTPLUS // linear units
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
        case GELU:
            return &GeLU;
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

type norm_mean_squared (type y, type ypred, int der){
    type dif = y - ypred;
    return der ? dif : .5 * dif * dif;
}

type ln (type in){ // log definition so that log(0) does not cause issues
    return in < exp(MIN_LOG_VAL) ? MIN_LOG_VAL : log(in);
}

type cross_entropy (type y, type ypred, int der){
    return der ? -(ypred/y) - (1-ypred)/(1-y): -ypred*ln(y) - (1-ypred)*ln(1-y);
}

type MAPE (type y, type ypred, int der){
    type dif = (y - ypred)/ypred;
    return der ? 100*(dif > 0 ? 1 : -1)/ypred : 100*abs(dif);
}


enum loss_func{ // classification for the loss functions
    MEAN_SQUARED,NMEAN_SQUARED,CROSS_ENTROPY,MAPD
};

type (*lfunc(loss_func function))(type, type, int){
    switch (function){
        case MEAN_SQUARED:
            return &mean_squared;
        case NMEAN_SQUARED:
            return &norm_mean_squared;
        case CROSS_ENTROPY:
            return &cross_entropy;
        case MAPD:
            return &MAPE;
        default:
            return &mean_squared;
    }
}