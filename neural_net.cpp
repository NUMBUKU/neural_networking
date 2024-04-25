# include <fstream>
# include <iostream>

# include "neuron.cpp"

using std::runtime_error,
    std::cout,
    std::ofstream;


class neural_net {
    protected:
        list dcdin;
        void change_previous (int layer, double dcda, double lr){ // recursive method for changing the previous activation
            list impact_list;
            for (int i = 0; i < net[layer].size(); i++){
                int wgtsize = net[layer][i].wgt.size();

                if (layer == 0) impact_list = calc_impact(in, net[layer][i], coef);
                else impact_list = calc_impact(midl[layer-1], net[layer][i], coef);

                net[layer][i].bias -= lr * dcda * impact_list[0];
                for (int j = 0; j < wgtsize; j++){
                    net[layer][i].wgt[j] -= lr * dcda * impact_list[j+1];
                    if (layer != 0) change_previous(layer-1, dcda * impact_list[j+wgtsize+1], lr);
                    else dcdin.push_back(impact_list[j+wgtsize+1]);
                }
            }
        }
    public:
        // variables for middle and output layers of the net
        Net net;
        Collumn outn;
        matrix midl;
        list outl,
        
        in; // variable for storing the given input of the net

        double certainty, // variable for storing the certainty of the net
        coef; // optional coefficient for certain activation functions

        int ins, // variable for storing the number of inputs of the net
        out = 0; // variable for storing the index of the highest value in neural_net::outl

        unsigned long iteration = 0; // variable for storing the number of feed forward cycles the net has gone through

        bool learn_coef, // boolean for storing if the user wants the net to learn the coefficients of the activation functions by itself
        initialised = false;

        void init (int inputs, int outputs, int layers, int rows, double coefficient = .01, bool learn_coefficients = false){ // initialiser
            ins = inputs;
            learn_coef = learn_coefficients;
            coef = coefficient;
            for (int i = 0; i < layers; i++){
                list actl (rows); Collumn collumn (rows);

                for (int j = 0; j < rows; j++) {
                    if (i == 0) for (int k = 0; k < ins; k++) collumn[j].wgt.push_back(1);
                    else for (int k = 0; k < rows; k++) collumn[j].wgt.push_back(1);
                    collumn[j].func = &SiLU;
                }
                net.push_back(collumn); midl.push_back(actl);
            }
            net.shrink_to_fit(); midl.shrink_to_fit();

            for (int i = 0; i < outputs; i++){
                neuron n;

                if (layers == 0) for (int k = 0; k < ins; k++) n.wgt.push_back(1);
                else for (int k = 0; k < rows; k++) n.wgt.push_back(1);
                n.out = 1;
                n.func = &normhyptan;

                outl.push_back(0); outn.push_back(n);
            }
            outl.shrink_to_fit(); outn.shrink_to_fit();

            for (int i = 0; i < inputs; i++)
                in.push_back(0);
            in.shrink_to_fit();

            dcdin = list (ins);

            initialised = true;
        }

        void print (){ // method for printing the net
            if (!initialised) throw runtime_error("Please run neural_net::init to initialise the net.");
            ofstream outputFile("data.txt"); // write data to a new file called "data.txt"

            for (int x = 0; x < net.size(); x++){
                for (int y = 0; y < net[x].size(); y++){
                    if (outputFile) outputFile << "name.net[" << x << ", " << y << "].bias = " <<  net[x][y].bias << ";\n";
                    cout << "name.net[" << x << ", " << y << "].bias = " << net[x][y].bias << ";\n";
                    for (int c = 0; c <  net[x][y].wgt.size(); c++){
                        if (outputFile) outputFile << "changeAtIndex(name.net[" << x << ", " << y << "].wgt, " << c << ", " <<  net[x][y].wgt[c] << ");\n";
                        cout << "changeAtIndex(name.net[" << x << ", " << y << "].wgt, " << c << ", " <<  net[x][y].wgt[c] << ");\n";
                    }
                }
            }
            for (int i = 0; i < outn.size(); i++){
                if (outputFile) outputFile << "name.outn[" << i << "].bias = " <<  outn[i].bias << ";\n";
                cout << "name.outn[" << i << "].bias = " <<  outn[i].bias << ";\n";
                for (int c = 0; c <  outn[i].wgt.size(); c++){
                    if (outputFile) outputFile << "changeAtIndex(name.outn[" << i << "].wgt, " << c << ", " <<  outn[i].wgt[c] << ");\n";
                    cout << "changeAtIndex(name.outn[" << i << "].wgt, " << c << ", " <<  outn[i].wgt[c] << ");\n";
                }
            }
        
            outputFile.close();
        }

        void modify (int inputs, int outputs, int layers, list rows, act_func functions []){ // resizes the net, can be non-rectangular (this will erase previous training data)
            if (!initialised) throw runtime_error("Please run neural_net::init to initialise the net.");
            if (/*(sizeof(functions)/sizeof(act_func)) != layers+1 || */rows.size() != layers) throw runtime_error("Rows list should be the same size as the amount of layers, and the size of the functions array should be one plus that size.");
            in.clear(); outl.clear(); outn.clear(); midl.clear(); net.clear(); 
            
            for (int i = 0; i < layers; i++){
                list actl (rows[i]); Collumn collumn (rows[i]);

                for (int j = 0; j < rows[i]; j++) {
                    if (i == 0) for (int k = 0; k < ins; k++) collumn[j].wgt[k] = 1;
                    else for (int k = 0; k < rows[i]; k++) collumn[j].wgt[k] = 1;
                    collumn[j].func = func(functions[i]);
                }
                
                net.push_back(collumn); midl.push_back(actl);
            }
            net.shrink_to_fit(); midl.shrink_to_fit();

            for (int i = 0; i < outputs; i++){
                neuron n; double d = 0;

                if (layers == 0) for (int k = 0; k < ins; k++) n.wgt[k] = 1;
                else for (int k = 0; k < rows[i]; k++) n.wgt[k] = 1;
                n.out = 1;
                n.func = func(functions[layers]);

                outl.push_back(d); outn.push_back(n);
            }
            outl.shrink_to_fit(); outn.shrink_to_fit();

            for (int i = 0; i < inputs; i++){
                double d = 0;
                in.push_back(d);
            }
            in.shrink_to_fit();
        }

        int calc_out (list input, bool normalise = false, list range = {0,0}){ // calculates the output of the net given an input
            if (!initialised) throw runtime_error("Please run neural_net::init to initialise the net.");
            if (input.size() != ins) throw runtime_error("Input list should be the same size as the amount of input neurons.");
            if (normalise && range.size() != 2) throw runtime_error("Range list should only have 2 values.");
            if (range[0] >= range[1]) throw runtime_error("First value of range list should be smaller than second value.");

            if (normalise) for (int i = 0; i < ins; i++) in;
            else in = input;
            
            for (int i = 0; i < net.size(); i++){
                for (int j = 0; j < net[i].size(); j++){
                    if (i == 0) midl[i][j] = calc_act(in, net[i][j], coef);
                    else midl[i][j] = calc_act(midl[i-1], net[i][j], coef);
                }
            }

            for (int i = 0; i < outn.size(); i++){
                if (net.size() == 0) outl[i] = calc_act(in, outn[i], coef);
                else outl[i] = calc_act(midl[net.size()-1], outn[i], coef);
            }

            double temp = outl[0];
            out = 0;
            double tot = 0;
            for (int i = 1; i <= outl.size(); i++){
                if (temp < outl[i]){
                    temp = outl[i];
                    out = i;
                }
                tot += outl[i];
            }
             
            certainty = 100*outl[out]/tot;

            iteration++;

            return out;
        }
    
        double calc_cost (list wanted){ // calculates the cost, a measure of how bad the net is performing (only works if neural_net::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_net::init to initialise the net.");
            int size = outl.size();
            if (wanted.size() != size) throw runtime_error("Wanted list shoulb be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_net::calc_out has already been run.");

            double cost = 0;
            for (int i = 0; i < size; i++) cost += loss(wanted[i], outl[i]);

            return cost;
        }

        void improve (list wanted, double learning_rate, int type = 0){ // improves the net by trying to lower the cost (only works if neural_net::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_net::init to initialise the net.");
            list impact_list;
            int size = net.size(),
            outsize = outn.size();

            if (wanted.size() != outsize) throw runtime_error("Wanted list shoulb be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_net::calc_out has already been run.");
            
            for (int i = 0; i < outsize; i++){
                int wgtsize = outn[i].wgt.size();
                if (size == 0) impact_list = calc_impact(in, outn[i], wanted[i], coef);
                else impact_list = calc_impact(midl[size-1], outn[i], wanted[i], coef);
                outn[i].bias -= learning_rate * impact_list[0];
                for (int j = 0; j < wgtsize; j++){
                    outn[i].wgt[j] -= learning_rate * impact_list[j+1];
                    if (size != 0) change_previous(size-1, impact_list[j+wgtsize+1], learning_rate);
                }
            }
        }
};


class CNN : neural_net {
    private:
        void printconv(){}
        double maxpool (matrix in){
            int rows = in[0].size();

            double returnval = 0;
            for (int i = 0; i < in.size(); i++) for (int j = 0; j < rows; j++) if (in[i][j] > returnval) returnval = in[i][j];

            return returnval;
        }
        
        double averagepool (matrix in){
            int rows = in[0].size(), col = in.size();

            double returnval = 0;
            for (int i = 0; i < col; i++) for (int j = 0; j < rows; j++) returnval += in[i][j];

            return returnval / (rows * col);
        }

        double conv(matrix I, matrix F){
            double returnval = 0;
            int rows = I[0].size();
            
            for (int i = 0; i < I.size(); i++) for (int j = 0; j < rows; j++) returnval += I[i][j] * F[i][j];

            return ReLU(returnval, 0, 0);
        }
    public:
        vector<vector<matrix>> K, // fourth order tensor for storing all the features
        channels; // fourth order tensor for storing all the convolutional channels

        list f;

        int xf, yf, sf,
        xpo, ypo, spo,
        conlay, xpa, ypa,
        xin, yin, xinff, yinff, inff;

        void init (int xinputs, int yinputs, int outputs, int fflayers, int ffrows, int conlayers, list features, double coef = .01, bool learn_coefficients = false, bool paddingvalid = true, int xfeature = 3, int yfeature = 3, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int stridefeature = 1){ // initialiser
            if (features.size() != conlayers) throw runtime_error("The size of the feature list should be the same as the amount of ");

            xin = xinputs, yin = yinputs;
            xinff = xinputs; yinff = yinputs;
            if (paddingvalid){
                xpa = (xfeature-1)/2;
                ypa = (yfeature-1)/2;
            } else {
                xpa = 0;
                ypa = 0;
            }

            for (int i = 0; i < conlay; i++){
                xinff = floor(1 + ((xinff+2*xpa - xfeature)/stridefeature));
                xinff = floor(1 + ((xinff - xpoolwindow)/stridepoolwindow));
                yinff = floor(1 + ((yinff+2*ypa - yfeature)/stridefeature));
                yinff = floor(1 + ((yinff - ypoolwindow)/stridepoolwindow));
            }
            double outchannels = features[0];
            for (int i = 1; i < conlay; i++){
                if (features[i] == 0) throw runtime_error("Number of features should be non-zero for every convolution layer.");
                outchannels *= features[i];
            }
            inff = outchannels * xinff * yinff;

            neural_net::init(inff, outputs, fflayers, ffrows, coef, learn_coefficients);

            xf = xfeature;
            yf = yfeature;
            sf = stridefeature;

            xpo = xpoolwindow;
            ypo = ypoolwindow;
            spo = stridepoolwindow;

            conlay = conlayers;

            f = features;

            for (int i = 0; i < conlay; i++){
                vector<matrix> tensor;

                for (int j = 0; j < features[i]; j++){
                    matrix kernell;
                    for (int k = 0; k < yf; k++){
                        list row;
                        for (int l = 0; l < xf; l++) row.push_back(1);
                        kernell.push_back(row);
                    }
                    tensor.push_back(kernell);
                }

                K.push_back(tensor);
            }
            
            int x = xin, y = yin, Nchan = 1;
            for (int i = 0; i < conlay*2 + 1; i++){
                channels.push_back(vector<matrix> (Nchan, matrix (y, list (x))));

                if (i % 2 == 0) { // convolution
                    x = floor(1 + ((x+2*xpa - xfeature)/stridefeature));
                    y = floor(1 + ((y+2*ypa - yfeature)/stridefeature));
                    Nchan *= features[i];
                } else { // pooling
                    x = floor(1 + ((x - xpoolwindow)/stridepoolwindow));
                    y = floor(1 + ((y - ypoolwindow)/stridepoolwindow));
                }
            }
        }

        void print (){ // method for printing the CNN
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init to initialise the net.");
            printconv(); neural_net::print();
        }

        void modify (){
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init to initialise the net.");
            
        }

        int calc_out (matrix input){
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init to initialise the net.");
            if (input.size() != yin || input[0].size() != xin) throw runtime_error("The input matrix should be of the size previously defined");
            
            matrix I (yf, list (xf)), in (ypo, list (xpo));
            list flattened (inff);

            // convolution
            channels[0][0] = input;

            int Nchan = -1;
            bool max = 1;
            for (int i = 1; i < conlay*2 + 1; i++){ // looping through convolution and pooling layers
                for (int j = 0; j < channels[i].size(); j++){ // looping through the channels
                    if (j % ((channels[i].size()/channels[i-1].size())) == 0) Nchan++;
                    for (int y = 0; y < channels[i][j].size(); y++) // looping through the channel
                        for (int x = 0; x < channels[i][j][y].size(); x++){
                            if ((i-1) % 2 == 0){ // convolution
                                for (int yI = 0; yI < yf; yI++) for (int xI = 0; xI < xf; xI++)
                                    I[yI][xI] = channels[i-1][Nchan][yI + y*sf][xI + x*sf];
                                channels[i][j][y][x] = conv(I, K[(i-1)/2][j]);
                            } else { // pooling
                                for (int yi = 0; yi < ypo; yi++) for (int xi = 0; xi < xpo; xi++)
                                    in[yi][xi] = channels[i-1][j][yi + y*spo][xi + x*spo];
                                if (max) channels[i][j][y][x] = maxpool(in);
                                else channels[i][j][y][x] = averagepool(in);

                                max = !max;
                            }
                        }
                }
            }

            // flattening
            for (int i = 0; i < inff /(yinff * xinff); i++) for (int j = 0; j < yinff; j++) for (int k = 0; k < xinff; k++)
                flattened.push_back(channels[conlay*2][i][j][k]);


            return neural_net::calc_out(flattened); // feedforward layers
        }

        void improve (list wanted, double learning_rate, int type = 0){
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init to initialise the net.");
            
            neural_net::improve(wanted, learning_rate, type);
            neural_net::dcdin;
        }
};

class RNN : neural_net {
    
};

class Transformer : neural_net {
    
};