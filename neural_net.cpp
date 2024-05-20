# include <fstream>
# include <iostream>

# include "neuron.cpp"

using std::runtime_error,
    std::cout,
    std::ofstream;


class neural_network {
    protected:
        list dcdin;
        void change_previous (int layer, double dcda, double lr, int batch_size){ // recursive method for changing the previous activation
            list impact_list;
            for (int i = 0; i < net[layer].size(); i++){
                int wgtsize = net[layer][i].wgt.size();

                if (layer == 0) impact_list = calc_impact(in, net[layer][i], actlist[lastlayer][i]);
                else impact_list = calc_impact(actlist[layer-1], net[layer][i], actlist[lastlayer][i]);

                net[layer][i].bias -= lr * dcda * impact_list[0] / batch_size;
                for (int j = 0; j < wgtsize; j++){
                    net[layer][i].wgt[j] -= lr * dcda * impact_list[j+1] / batch_size;
                    if (layer != 0) change_previous(layer-1, dcda * impact_list[j+wgtsize+1], lr, batch_size);
                    else dcdin[j] = impact_list[j+wgtsize+1];
                }
            }
        }
    public:
        // variables for middle and output layers of the net
        Net net;
        matrix actlist;
        
        list in, // variable for storing the given input of the net
        outlist; // variable for storing the output list of the net (the same as actlist[lastlayer])

        double certainty; // variable for storing the certainty of the net

        int out_index = 0, // variable for storing the index of the highest value in neural_network::outlist
        lastlayer = -1; // variable for storing the index of the last layer

        unsigned long iteration = 0; // variable for storing the number of feed forward cycles the net has gone through

        bool learn_coeffficents, // boolean for storing if the user wants the net to learn the coefficients of the activation functions by itself
        initialised = false;

        void add_dense_layer (int neuron_count, act_func activation_function, int input_count = 0, double coefficient = 0.01){            
            Collumn layer (neuron_count);
            for (int i = 0; i < neuron_count; i++){
                if (net.size() != 0) layer[i].wgt = list (net[lastlayer].size(), 1);
                else {
                    if (input_count <= 0) throw runtime_error("For the first layer, input_count should be defined as a number greater than 0.");
                    layer[i].wgt = list (input_count, 1);
                    initialised = true;
                }

                layer[i].coef = coefficient;

                if (activation_function != SOFTMAX) layer[i].func = func(activation_function);
                else layer[i].softmax = true;
            }
            net.push_back(layer); net.shrink_to_fit();

            actlist.push_back(list (neuron_count, 0)); actlist.shrink_to_fit();

            lastlayer++;
        }

        void export_net (bool write_to_terminal = true, bool write_to_file = false){ // method for printing the net
            if (!initialised) throw runtime_error("Please run neural_network::add_dense_layer to initialise the net.");

            if (!write_to_file && !write_to_terminal) cout << "ok...";

            ofstream outputFile;
            if (write_to_file) outputFile = ofstream ("data.txt"); // write data to a new file called "data.txt"

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
            for (int i = 0; i < net[lastlayer].size(); i++){
                if (outputFile && write_to_file) outputFile << "name.net[lastlayer][" << i << "].bias = " <<  net[lastlayer][i].bias << ";\n";
                if (write_to_terminal) cout << "name.net[lastlayer][" << i << "].bias = " <<  net[lastlayer][i].bias << ";\n";
                for (int c = 0; c < net[lastlayer][i].wgt.size(); c++){
                    if (outputFile && write_to_file) outputFile << "name.net[lastlayer][" << i << "].wgt[" << c << "] = " <<  net[lastlayer][i].wgt[c] << ";\n";
                    if (write_to_terminal) cout << "name.net[lastlayer][" << i << "].wgt[" << c << "] = " <<  net[lastlayer][i].wgt[c] << ";\n";
                }
            }
        
            outputFile.close();
        }

        int evaluate (list input){ // calculates the output of the net given an input
            if (!initialised) throw runtime_error("Please run neural_network::add_dense_layer to initialise the net.");
            if (input.size() != net[0][0].wgt.size()) throw runtime_error("Input list should be the same size as the amount of input neurons.");
            
            in = input;

            for (int i = 0; i < net.size() - 1; i++){
                for (int j = 0; j < net[i].size(); j++){
                    if (i == 0) actlist[i][j] = calc_act(in, net[i][j]);
                    else actlist[i][j] = calc_act(actlist[i-1], net[i][j]);
                }
            }

            for (int i = 0; i < net[lastlayer].size(); i++){
                if (net.size() == 1) actlist[lastlayer][i] = calc_act(in, net[lastlayer][i]);
                else actlist[lastlayer][i] = calc_act(actlist[net.size()-1], net[lastlayer][i]);
            }
            
            double temp = actlist[lastlayer][0],
            tot = abs(actlist[lastlayer][0]);
            out_index = 0;
            for (int i = 1; i < actlist[lastlayer].size(); i++){
                if (temp < actlist[lastlayer][i]){
                    temp = actlist[lastlayer][i];
                    out_index = i;
                }
                tot += abs(actlist[lastlayer][i]);
            }
             
            certainty = 100*(abs(actlist[lastlayer][out_index])/tot);

            outlist = actlist[lastlayer];

            iteration++;
            return out_index;
        }
    
        double calc_loss (list wanted, loss_func function = MEAN_SQUARED){ // calculates the cost, a measure of how bad the net is performing (only works if neural_network::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_network::add_dense_layer to initialise the net.");
            int size = actlist[lastlayer].size();
            if (wanted.size() != size) throw runtime_error("Wanted list shoulb be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_network::evaluate has already been run.");

            double cost = 0;
            for (int i = 0; i < size; i++) cost += lfunc(function)(wanted[i], actlist[lastlayer][i], 0);

            return cost;
        }

        void fit (list wanted, double learning_rate, int batch_size = 1, loss_func function = MEAN_SQUARED){ // improves the net by trying to lower the cost (only works if neural_network::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_network::add_dense_layer to initialise the net.");
            list impact_list;
            int size = net.size(),
            outsize = net[lastlayer].size();

            if (wanted.size() != outsize) throw runtime_error("Wanted list should be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_network::evaluate has already been run.");
            
            for (int i = 0; i < outsize; i++){
                int wgtsize = net[lastlayer][i].wgt.size();

                if (size == 0) impact_list = calc_impact(in, net[lastlayer][i], wanted[i], actlist[lastlayer][i], function);
                else impact_list = calc_impact(actlist[size-1], net[lastlayer][i], wanted[i], actlist[lastlayer][i], function);

                net[lastlayer][i].bias -= (learning_rate * impact_list[0] / batch_size);

                for (int j = 0; j < wgtsize; j++){
                    net[lastlayer][i].wgt[j] -= (learning_rate * impact_list[j+1] / batch_size);
                    if (size != 1) change_previous(size-1, impact_list[j+wgtsize+1], learning_rate, batch_size);
                }
            }
        }
};


class CNN : public neural_network {
    private:
        void printconv(bool term, bool file){

        }

        double pool (matrix in, bool m){
            int rows = in[0].size(), col = in.size();
            double returnval = 0;

            for (int i = 0; i < col; i++) for (int j = 0; j < rows; j++){
                if (in[i][j] > returnval && m) returnval = in[i][j];
                if (!m) returnval += in[i][j];
            }

            return m ? returnval : returnval / (rows * col);
        }

        matrix backpool (matrix in, double out, bool m){
            int rows = in[0].size(), col = in.size();
            matrix returnval;
            if (m) returnval = matrix(col, list (rows, 0));
            else {returnval = matrix(col, list (rows, 1/(rows*col))); return returnval;}

            for (int i = 0; i < col; i++) for (int j = 0; j < rows; j++)
                if (m && in[i][j] == out) {returnval[i][j] = 1; break;}
            

            return returnval;
        }

        double conv(matrix I, matrix F){
            double returnval = 0;
            int rows = I[0].size();
            
            for (int i = 0; i < I.size(); i++) for (int j = 0; j < rows; j++) returnval += I[i][j] * F[i][j];

            return ReLU(returnval, 0, 0);
        }

    public:
        vector<vector<matrix>> K, // fourth order tensor for storing all the features
        channels, // fourth order tensor for storing all the convolutional channels
        backprop; // fourth order tensor for storing all the  partial derivatives

        matrix biasses; // matrix for storing biasses of convolution layers

        vector<int> f;

        int xf, yf, sf,
        xpo, ypo, spo,
        conlay, xpa, ypa,
        xin, yin, xinANN, yinANN, inANN,
        outchan;

        vector<bool> max;

        void init (int xinputs, int yinputs, int conlayers, int features, bool paddingvalid = true, int xfeature = 3, int yfeature = 3, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int stridefeature = 1){ // initialiser
            xin = xinputs, yin = yinputs;
            xinANN = xinputs; yinANN = yinputs;
            xf = xfeature; yf = yfeature; sf = stridefeature;

            xpo = xpoolwindow; ypo = ypoolwindow; spo = stridepoolwindow;

            if (paddingvalid){xpa = (xfeature-1)/2; ypa = (yfeature-1)/2;}
            else {xpa = 0; ypa = 0;}

            conlay = conlayers;

            f = vector<int>(conlayers, features);
            
            // initialising feedforward layers, biasses and pooltypes
            bool m = false;
            for (int i = 0; i < conlay; i++){
                xinANN = floor(1 + ((xinANN+2*xpa - xfeature)/stridefeature));
                xinANN = floor(1 + ((xinANN - xpoolwindow)/stridepoolwindow));
                yinANN = floor(1 + ((yinANN+2*ypa - yfeature)/stridefeature));
                yinANN = floor(1 + ((yinANN - ypoolwindow)/stridepoolwindow));

                biasses.push_back(list(channels[(i*2)+1].size(), 0));

                max.push_back(m);
                m = !m;
            }
            biasses.shrink_to_fit();

            outchan = pow(features, conlay);
            inANN = outchan * xinANN * yinANN;
            
            K = vector<vector<matrix>> (conlay, vector<matrix> (features, matrix (yf, list (xf, 1))));
            
            int x = xin + 2*xpa, y = yin + 2* ypa, Nchan = 1;
            for (int i = 1; i < conlay*2 + 1; i++){
                channels.push_back(vector<matrix> (Nchan, matrix (y, list (x, 0)))); // create all channels for one layer, filled with zeroes for padding

                if ((i-1) % 2 == 0) { // convolution
                    x = floor(1 + ((x - xfeature)/stridefeature))+2*xpa;
                    y = floor(1 + ((y - yfeature)/stridefeature))+2*ypa;
                    Nchan *= features;
                } else { // pooling
                    x = floor(1 + ((x - xpoolwindow)/stridepoolwindow));
                    y = floor(1 + ((y - ypoolwindow)/stridepoolwindow));
                }
            }
            channels.shrink_to_fit();

            x = xin; y = yin; Nchan = 1;
            for (int i = 1; i < conlay*2 + 1; i++){
                if (i == 0) continue;
                backprop.push_back(vector<matrix> (Nchan, matrix (y, list (x, 0)))); // create all channels for one layer, filled with zeroes for padding

                if ((i-1) % 2 == 0) { // convolution
                    x = floor(1 + ((x+2*xpa - xfeature)/stridefeature));
                    y = floor(1 + ((y+2*ypa - yfeature)/stridefeature));
                    Nchan *= features;
                } else { // pooling
                    x = floor(1 + ((x - xpoolwindow)/stridepoolwindow));
                    y = floor(1 + ((y - ypoolwindow)/stridepoolwindow));
                }
            }
            backprop.shrink_to_fit();
        }

        void add_dense_layer (int neuron_count, act_func activation_function, double coefficient = 0.01){
            Collumn layer (neuron_count);
            for (int i = 0; i < neuron_count; i++){
                if (net.size() != 0) layer[i].wgt = list (net[lastlayer].size(), 1);
                else layer[i].wgt = list (inANN, 1);

                layer[i].coef = coefficient;

                if (activation_function != SOFTMAX) layer[i].func = func(activation_function);
                else layer[i].softmax = true;
            }
            net.push_back(layer); net.shrink_to_fit();

            actlist.push_back(list (neuron_count, 0)); actlist.shrink_to_fit();

            lastlayer++;
        }

        void export_net (bool write_to_terminal = true, bool write_to_file = false){ // method for printing the CNN
            if (!neural_network::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            printconv(write_to_terminal, write_to_file); neural_network::export_net(write_to_terminal, write_to_file);
        }

        // void modify (int xinputs, int yinputs, int outputs, int fflayers, vector<int> ffrows, int conlayers, vector<int> features, vector<bool> pooltype, act_func functions [], double coef = .01, bool learn_coefficients = false, bool paddingvalid = true, int xfeature = 3, int yfeature = 3, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int stridefeature = 1){
        //     if (!neural_network::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
        //     if (features.size() != conlayers || pooltype.size()) throw runtime_error("The size of the feature and pooltype list should be the same as the amount of convolutional layers");

        //     matrix().swap(biasses); vector<vector<matrix>>().swap(K); vector<vector<matrix>>().swap(channels);
            
        //     xin = xinputs, yin = yinputs;
        //     xinANN = xinputs; yinANN = yinputs;
        //     xf = xfeature; yf = yfeature; sf = stridefeature;

        //     xpo = xpoolwindow; ypo = ypoolwindow; spo = stridepoolwindow;

        //     if (paddingvalid){xpa = (xfeature-1)/2; ypa = (yfeature-1)/2;}
        //     else {xpa = 0; ypa = 0;}

        //     conlay = conlayers;

        //     f = features;
            
        //     // initialising feedforward layers
        //     for (int i = 0; i < conlay; i++){
        //         xinANN = floor(1 + ((xinANN+2*xpa - xfeature)/stridefeature));
        //         xinANN = floor(1 + ((xinANN - xpoolwindow)/stridepoolwindow));
        //         yinANN = floor(1 + ((yinANN+2*ypa - yfeature)/stridefeature));
        //         yinANN = floor(1 + ((yinANN - ypoolwindow)/stridepoolwindow));

        //         biasses.push_back(list(channels[(i*2)+1].size(), 0));
        //     }
        //     biasses.shrink_to_fit();

        //     int outchannels = 1;
        //     for (int i = 0; i < conlay; i++){
        //         if (f[i] <= 0) throw runtime_error("All elements of the features list should be above zero.");
        //         outchannels *= f[i];
        //     }
        //     inANN = outchannels * xinANN * yinANN;
        //     neural_network::modify(inANN, outputs, fflayers, ffrows, functions, coef, learn_coefficients);

        //     //K = vector<vector<matrix>> (conlay, vector<matrix> (features, matrix (yf, list (xf, 1))));
        //     for (int i = 0; i < conlay; i++) K.push_back(vector<matrix> (f[i], matrix (yf, list(xf, 1))));
            
        //     int x = xin, y = yin, Nchan = 1;
        //     for (int i = 1; i < conlay*2 + 1; i++){
        //         channels.push_back(vector<matrix> (Nchan, matrix (y, list (x, 0)))); // create all channels for one layer, filled with zeroes for padding

        //         if ((i-1) % 2 == 0) { // convolution
        //             x = floor(1 + ((x+2*xpa - xfeature)/stridefeature));
        //             y = floor(1 + ((y+2*ypa - yfeature)/stridefeature));
        //             Nchan *= f[(i-1)/2];
        //         } else { // pooling
        //             x = floor(1 + ((x - xpoolwindow)/stridepoolwindow));
        //             y = floor(1 + ((y - ypoolwindow)/stridepoolwindow));
        //         }
        //     }
        //     channels.shrink_to_fit();
            
        //     max = pooltype;
        // }

        int calc_out (matrix input){
            if (!neural_network::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            
            // convolution and pooling layers
            matrix I (yf, list (xf)), in (ypo, list (xpo)),
            padded;
            list flattened (inANN);

            if (insmatrix(channels[0][0], input, xpa, ypa)) throw runtime_error("The input matrix should be of the size previously defined");

            int Nchan = -1;
            for (int i = 1; i < conlay*2 + 1; i++){ // looping through convolution and pooling layers
                int truelay = floor((i-1)/2);
                int ch = channels[i].size();
                for (int j = 0; j < ch; j++){ // looping through the channels
                    int yim = channels[i][j].size();

                    if (j % ((ch/channels[i-1].size())) == 0) Nchan++; // selecting which channel is being convoluted
                    if ((i-1) % 2 == 0) {
                        int ys = channels[i-1][Nchan].size(), xs = channels[i-1][Nchan][0].size();
                        padded = matrix (ys + 2*ypa, list (xs + 2 * xpa, 0));

                        for (int y = 0; y < ys; y++) for (int x = 0; x < xs; x++){
                            padded[y+ypa][x+xpa] = channels[i-1][Nchan][y][x];
                        }
                    } else matrix().swap(padded);

                    for (int y = 0; y < yim; y++){ // looping through the channel matrix
                        int xim = channels[i][j][y].size();

                        for (int x = 0; x < xim; x++){

                            if ((i-1) % 2 == 0){ // convolution
                                for (int yI = 0; yI < yf; yI++) for (int xI = 0; xI < xf; xI++)
                                    I[yI][xI] = padded[yI + y*sf][xI + x*sf];
                                channels[i][j][y][x] = conv(I, K[truelay][j]) + biasses[truelay][j];
                            } 
                            
                            else { // pooling
                                for (int yi = 0; yi < ypo; yi++) for (int xi = 0; xi < xpo; xi++)
                                    in[yi][xi] = channels[i-1][Nchan][yi + y*spo][xi + x*spo];
                                
                                channels[i][j][y][x] = pool(in, max[truelay]);
                            }
                        }
                    }
                }
                Nchan = -1;
            }

            // flattening
            for (int i = 0; i < outchan; i++) for (int j = 0; j < yinANN; j++) for (int k = 0; k < xinANN; k++)
                flattened.push_back(channels[conlay*2][i][j][k]);
            flattened.shrink_to_fit();

            return neural_network::evaluate(flattened); // feedforward layers
        }

        void fit (list wanted, double learning_rate, int batch_size = 1){
            if (!neural_network::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            
            neural_network::fit(wanted, learning_rate);

            int index = 0;           
            for (int i = 0; i < outchan; i++) for (int j = 0; j < yinANN; j++) for (int k = 0; k < xinANN; k++){
                backprop[conlay*2][i][j][k] = dcdin[index];
                index++;
            }
            for (int lay = conlay*2; lay > 0; lay--){ // propagating backwards through convolutional layers
                int truelay = floor(lay/2 + 1);
                for (int c = 0; c < backprop[lay].size(); c++)
                    for (int y = 0; y < backprop[lay][c].size(); y++)
                        for (int x = 0; x < backprop[lay][c][y].size(); x++){
                            if (lay % 2 == 0){ // convolution

                            } else { // pooling
                                //backprop[lay - 1][c] = backpool();
                            }
                        }
            }
        }
};

class RNN : public neural_network {
    private:
    public:
        void init (){

        }
};

class LSTM : public neural_network {
    
};