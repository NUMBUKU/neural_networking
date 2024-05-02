# include <fstream>
# include <iostream>

# include "neuron.cpp"

using std::runtime_error,
    std::cout,
    std::ofstream;


class neural_net {
    protected:
        list dcdin;
        void change_previous (int layer, double dcda, double lr, int batch_size, int type){ // recursive method for changing the previous activation
            list impact_list;
            for (int i = 0; i < net[layer].size(); i++){
                int wgtsize = net[layer][i].wgt.size();

                if (layer == 0) impact_list = calc_impact(in, net[layer][i], coef);
                else impact_list = calc_impact(midl[layer-1], net[layer][i], coef);

                net[layer][i].bias -= lr * dcda * impact_list[0] / batch_size;
                for (int j = 0; j < wgtsize; j++){
                    net[layer][i].wgt[j] -= lr * dcda * impact_list[j+1] / batch_size;
                    if (layer != 0) change_previous(layer-1, dcda * impact_list[j+wgtsize+1], lr, batch_size, type);
                    else dcdin[j] = impact_list[j+wgtsize+1];
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

        void init (int inputs, int outputs, int layers, int rows, act_func middle_act_function = LRELU, act_func out_act_func = NTANH, double coefficient = .01, bool learn_coefficients = false){ // initialiser
            ins = inputs;
            learn_coef = learn_coefficients;
            coef = coefficient;
            for (int i = 0; i < layers; i++){
                list actl (rows); Collumn collumn (rows);

                for (int j = 0; j < rows; j++) {
                    if (i == 0) collumn[j].wgt = list (ins, 1);
                    else collumn[j].wgt = list (rows, 1);
                    collumn[j].func = func(middle_act_function);
                }
                net.push_back(collumn); midl.push_back(actl);
            }
            net.shrink_to_fit(); midl.shrink_to_fit();
            
            outl = list (outputs);
            for (int i = 0; i < outputs; i++){
                neuron n;

                if (layers == 0) n.wgt = list (ins, 1);
                else n.wgt = list (rows, 1);
                
                n.out = 1;
                n.func = func(out_act_func);

                outn.push_back(n);
            }
            outn.shrink_to_fit();

            in = list (ins);

            dcdin = list (ins);

            initialised = true;
        }

        void print (bool write_to_terminal = true, bool write_to_file = false){ // method for printing the net
            if (!initialised) throw runtime_error("Please run neural_net::init or neural_net::modify to initialise the net.");

            if (!write_to_file && !write_to_terminal) cout << "ok...";

            ofstream outputFile("data.txt"); // write data to a new file called "data.txt"

            for (int x = 0; x < net.size(); x++){
                for (int y = 0; y < net[x].size(); y++){
                    if (outputFile && write_to_file) outputFile << "name.net[" << x << "][" << y << "].bias = " <<  net[x][y].bias << ";\n";
                    if (write_to_terminal) cout << "name.net[" << x << "][" << y << "].bias = " << net[x][y].bias << ";\n";
                    for (int c = 0; c <  net[x][y].wgt.size(); c++){
                        if (outputFile && write_to_file) outputFile << "name.net[" << x << "][" << y << "].wgt[" << c << "] = " <<  net[x][y].wgt[c] << ";\n";
                        if (write_to_terminal) cout << "name.net[" << x << "][" << y << "].wgt[" << c << "] = " <<  net[x][y].wgt[c] << ";\n";
                    }
                }
            }
            for (int i = 0; i < outn.size(); i++){
                if (outputFile && write_to_file) outputFile << "name.outn[" << i << "].bias = " <<  outn[i].bias << ";\n";
                if (write_to_terminal) cout << "name.outn[" << i << "].bias = " <<  outn[i].bias << ";\n";
                for (int c = 0; c <  outn[i].wgt.size(); c++){
                    if (outputFile && write_to_file) outputFile << "name.outn[" << i << "].wgt[" << c << "] = " <<  outn[i].wgt[c] << ";\n";
                    if (write_to_terminal) cout << "name.outn[" << i << "].wgt[" << c << "] = " <<  outn[i].wgt[c] << ";\n";
                }
            }
        
            outputFile.close();
        }

        void modify (int inputs, int outputs, int layers, vector<int> rows, act_func functions [], double coefficient = .01, bool learn_coefficients = false){ // resizes the net, can be non-rectangular (this will erase previous training data)
            if (rows.size() != layers) throw runtime_error("Rows list should be the same size as the amount of layers.");
            
            list().swap(in); list().swap(outl); Collumn().swap(outn); matrix().swap(midl); Net().swap(net);

            ins = inputs;
            learn_coef = learn_coefficients;
            coef = coefficient;
            
            for (int i = 0; i < layers; i++){
                list actl (rows[i]); Collumn collumn (rows[i]);

                for (int j = 0; j < rows[i]; j++) {
                    if (i == 0) collumn[j].wgt = list (ins, 1);
                    else collumn[j].wgt = list (rows[i-1], 1);
                    try {
                        collumn[j].func = func(functions[i]);
                    } catch (std::out_of_range){
                        throw runtime_error("The size of the functions array should be one plus the amount of layers.");
                    }
                }
                
                net.push_back(collumn); midl.push_back(actl);
            }
            net.shrink_to_fit(); midl.shrink_to_fit();

            outl = list (outputs);
            for (int i = 0; i < outputs; i++){
                neuron n;

                if (layers == 0) n.wgt = list (ins, 1);
                else n.wgt = list (ins, 1);
                n.out = 1;
                n.func = func(functions[layers]);

                outn.push_back(n);
            }
            outn.shrink_to_fit();

            in = list (ins);

            dcdin = list (ins);

            initialised = true;
        }

        int calc_out (list input, bool normalise = false, list range = {0,1}){ // calculates the output of the net given an input
            if (!initialised) throw runtime_error("Please run neural_net::init or neural_net::modify to initialise the net.");
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
            for (int i = 0; i <= outl.size(); i++){
                if (temp < outl[i]){
                    temp = outl[i];
                    out = i;
                }
                tot += abs(outl[i]);
            }
             
            certainty = 100*(abs(outl[out])/tot);

            iteration++;

            return out;
        }
    
        double calc_loss (list wanted){ // calculates the cost, a measure of how bad the net is performing (only works if neural_net::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_net::init or neural_net::modify to initialise the net.");
            int size = outl.size();
            if (wanted.size() != size) throw runtime_error("Wanted list shoulb be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_net::calc_out has already been run.");

            double cost = 0;
            for (int i = 0; i < size; i++) cost += singleloss(wanted[i], outl[i]);

            return cost;
        }

        void improve (list wanted, double learning_rate, int batch_size = 1, int type = 0){ // improves the net by trying to lower the cost (only works if neural_net::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_net::init or neural_net::modify to initialise the net.");
            list impact_list;
            int size = net.size(),
            outsize = outn.size();

            if (wanted.size() != outsize) throw runtime_error("Wanted list shoulb be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_net::calc_out has already been run.");
            
            for (int i = 0; i < outsize; i++){
                int wgtsize = outn[i].wgt.size();
                if (size == 0) impact_list = calc_impact(in, outn[i], wanted[i], coef);
                else impact_list = calc_impact(midl[size-1], outn[i], wanted[i], coef);
                outn[i].bias -= learning_rate * impact_list[0] / batch_size;
                for (int j = 0; j < wgtsize; j++){
                    outn[i].wgt[j] -= learning_rate * impact_list[j+1] / batch_size;
                    if (size != 0) change_previous(size-1, impact_list[j+wgtsize+1], learning_rate, batch_size, type);
                }
            }
        }
};


class CNN : public neural_net {
    private:
        void printconv(bool term, bool file){

        }

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

        matrix biasses; // matrix for storing biasses of convolution layers

        vector<int> f;

        int xf, yf, sf,
        xpo, ypo, spo,
        conlay, xpa, ypa,
        xin, yin, xinff, yinff, inff,
        outchan;

        void init (int xinputs, int yinputs, int outputs, int fflayers, int ffrows, int conlayers, int features, act_func middle_act_function = LRELU, act_func out_act_func = NTANH, double coef = .01, bool learn_coefficients = false, bool paddingvalid = true, int xfeature = 3, int yfeature = 3, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int stridefeature = 1){ // initialiser
            xin = xinputs, yin = yinputs;
            xinff = xinputs; yinff = yinputs;
            xf = xfeature; yf = yfeature; sf = stridefeature;

            xpo = xpoolwindow; ypo = ypoolwindow; spo = stridepoolwindow;

            if (paddingvalid){xpa = (xfeature-1)/2; ypa = (yfeature-1)/2;}
            else {xpa = 0; ypa = 0;}

            conlay = conlayers;

            f = vector<int>(fflayers, features);
            
            // initialising feedforward layers
            for (int i = 0; i < conlay; i++){
                xinff = floor(1 + ((xinff+2*xpa - xfeature)/stridefeature));
                xinff = floor(1 + ((xinff - xpoolwindow)/stridepoolwindow));
                yinff = floor(1 + ((yinff+2*ypa - yfeature)/stridefeature));
                yinff = floor(1 + ((yinff - ypoolwindow)/stridepoolwindow));
            }
            outchan = pow(features, conlay);
            inff = outchan * xinff * yinff;
            neural_net::init(inff, outputs, fflayers, ffrows, middle_act_function, out_act_func, coef, learn_coefficients);

            K = vector<vector<matrix>> (conlay, vector<matrix> (features, matrix (yf, list (xf, 1))));
            
            int x = xin, y = yin, Nchan = 1;
            for (int i = 1; i < conlay*2 + 1; i++){
                channels.push_back(vector<matrix> (Nchan, matrix (y, list (x, 0)))); // create all channels for one layer, filled with zeroes for padding

                if ((i-1) % 2 == 0) { // convolution
                    x = floor(1 + ((x+2*xpa - xfeature)/stridefeature));
                    y = floor(1 + ((y+2*ypa - yfeature)/stridefeature));
                    Nchan *= features;
                } else { // pooling
                    x = floor(1 + ((x - xpoolwindow)/stridepoolwindow));
                    y = floor(1 + ((y - ypoolwindow)/stridepoolwindow));
                }
            }
            channels.shrink_to_fit();

            for (int i = 0; i < conlay; i++) biasses.push_back(list(channels[(i*2)+1].size(), 0));
            biasses.shrink_to_fit();
        }

        void print (bool write_to_terminal = true, bool write_to_file = false){ // method for printing the CNN
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            printconv(write_to_terminal, write_to_file); neural_net::print(write_to_terminal, write_to_file);
        }

        void modify (int xinputs, int yinputs, int outputs, int fflayers, vector<int> ffrows, int conlayers, vector<int> features, vector<int> pooltype, act_func functions [], double coef = .01, bool learn_coefficients = false, bool paddingvalid = true, int xfeature = 3, int yfeature = 3, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int stridefeature = 1){
            if (features.size() != conlayers) throw runtime_error("The size of the feature list should be the same as the amount of convolutional layers");
            
            matrix().swap(biasses); vector<vector<matrix>>().swap(K); vector<vector<matrix>>().swap(channels);
            
            xin = xinputs, yin = yinputs;
            xinff = xinputs; yinff = yinputs;
            xf = xfeature; yf = yfeature; sf = stridefeature;

            xpo = xpoolwindow; ypo = ypoolwindow; spo = stridepoolwindow;

            if (paddingvalid){xpa = (xfeature-1)/2; ypa = (yfeature-1)/2;}
            else {xpa = 0; ypa = 0;}

            conlay = conlayers;

            f = features;
            
            // initialising feedforward layers
            for (int i = 0; i < conlay; i++){
                xinff = floor(1 + ((xinff+2*xpa - xfeature)/stridefeature));
                xinff = floor(1 + ((xinff - xpoolwindow)/stridepoolwindow));
                yinff = floor(1 + ((yinff+2*ypa - yfeature)/stridefeature));
                yinff = floor(1 + ((yinff - ypoolwindow)/stridepoolwindow));
            }
            int outchannels = 1;
            for (int i = 0; i < conlay; i++){
                if (f[i] <= 0) throw runtime_error("All elements of the features list should be above zero.");
                outchannels *= f[i];
            }
            inff = outchannels * xinff * yinff;
            neural_net::modify(inff, outputs, fflayers, ffrows, functions, coef, learn_coefficients);

            //K = vector<vector<matrix>> (conlay, vector<matrix> (features, matrix (yf, list (xf, 1))));
            for (int i = 0; i < conlay; i++) K.push_back(vector<matrix> (f[i], matrix (yf, list(xf, 1))));
            
            int x = xin, y = yin, Nchan = 1;
            for (int i = 1; i < conlay*2 + 1; i++){
                channels.push_back(vector<matrix> (Nchan, matrix (y, list (x, 0)))); // create all channels for one layer, filled with zeroes for padding

                if ((i-1) % 2 == 0) { // convolution
                    x = floor(1 + ((x+2*xpa - xfeature)/stridefeature));
                    y = floor(1 + ((y+2*ypa - yfeature)/stridefeature));
                    Nchan *= f[(i-1)/2];
                } else { // pooling
                    x = floor(1 + ((x - xpoolwindow)/stridepoolwindow));
                    y = floor(1 + ((y - ypoolwindow)/stridepoolwindow));
                }
            }
            channels.shrink_to_fit();

            for (int i = 0; i < conlay; i++) biasses.push_back(list(channels[(i*2)+1].size(), 0));
            biasses.shrink_to_fit();
        }

        int calc_out (matrix input){
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            if (input.size() != yin || input[0].size() != xin) throw runtime_error("The input matrix should be of the size previously defined");
            
            // convolution and pooling layers
            matrix I (yf, list (xf)), in (ypo, list (xpo)),
            padded;
            list flattened (inff);

            channels[0][0] = input;

            int Nchan = -1;
            bool max = false;
            for (int i = 1; i < conlay*2 + 1; i++){ // looping through convolution and pooling layers
                int ch = channels[i].size();
                for (int j = 0; j < ch; j++){ // looping through the channels
                    int yim = channels[i][j].size();

                    if (j % ((ch/channels[i-1].size())) == 0) Nchan++; // selecting which channel is being convoluted
                    if ((i-1) % 2 == 0) padded = matrix (channels[i-1][Nchan].size()+ 2*ypa, list (channels[i-1][Nchan][0].size(), 0));

                    for (int y = 0; y < yim; y++){ // looping through the channel matrix
                        int xim = channels[i][j][y].size();

                        for (int x = 0; x < xim; x++){

                            if ((i-1) % 2 == 0){ // convolution
                                for (int yI = 0; yI < yf; yI++) for (int xI = 0; xI < xf; xI++)
                                    I[yI][xI] = padded[yI + y*sf][xI + x*sf];
                                channels[i][j][y][x] = conv(I, K[(i-1)/2][j]) + biasses[(i-1)/2][j];
                            } 
                            
                            else { // pooling
                                for (int yi = 0; yi < ypo; yi++) for (int xi = 0; xi < xpo; xi++)
                                    in[yi][xi] = channels[i-1][Nchan][yi + y*spo][xi + x*spo];
                                
                                if (max) channels[i][j][y][x] = maxpool(in);
                                else channels[i][j][y][x] = averagepool(in);

                                max = !max;
                            }
                        }
                    }
                }
                Nchan = -1;
            }

            // flattening
            for (int i = 0; i < inff / (yinff * xinff); i++) for (int j = 0; j < yinff; j++) for (int k = 0; k < xinff; k++)
                flattened.push_back(channels[conlay*2][i][j][k]);


            return neural_net::calc_out(flattened); // feedforward layers
        }

        void improve (list wanted, double learning_rate, int type = 0){
            if (!neural_net::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            
            neural_net::improve(wanted, learning_rate, type);
            neural_net::dcdin;
        }
};

class RNN : public neural_net {
    
};

// class Transformer : public neural_net {
    
// };