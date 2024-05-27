# include <fstream>
# include <iostream>

# include "neuron.cpp"

using std::runtime_error,
    std::cout,
    std::ofstream;


class neural_network {
    protected:
        list dcdin;
        void change_previous (int layer, type dcda, type lr, int batch_size){ // recursive method for changing the previous activation
            list impact_list;
            for (int i = 0; i < net[layer].size(); i++){
                int wgtsize = net[layer][i].wgt.size();

                if (layer == 0) impact_list = calc_impact(in, &net[layer][i], actlist[lastlayer][i], learn_coeffficents);
                else impact_list = calc_impact(actlist[layer-1], &net[layer][i], actlist[lastlayer][i], learn_coeffficents);

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

        type certainty; // variable for storing the certainty of the net

        int out_index = 0, // variable for storing the index of the highest value in neural_network::outlist
        lastlayer = -1, // variable for storing the index of the last layer
        incount;

        unsigned long long iteration = 0; // variable for storing the number of feed forward cycles the net has gone through

        bool learn_coeffficents, // boolean for storing if the user wants the net to learn the coefficients of the activation functions by itself
        initialised = false,
        inputinitialised = false;

        void add_input (int input_count){
            if (input_count <= 0) throw runtime_error("input_count should be greater than zero.");
            if (inputinitialised) throw runtime_error("Input was already defined.");

            incount = input_count;
            
            inputinitialised = true;
        }

        void add_dense_layer (int neuron_count, act_func activation_function, type coefficient = 0.01){            
            Collumn layer (neuron_count);
            if (inputinitialised) throw runtime_error("neural_network::add_input should have been run before adding dense layers.");
            for (int i = 0; i < neuron_count; i++){
                if (net.size() == 0) list (net[lastlayer].size(), 1);
                else layer[i].wgt = list (incount, 1);

                layer[i].coef = coefficient;

                if (activation_function != SOFTMAX) layer[i].func = func(activation_function);
                else layer[i].softmax = true;
            }
            net.push_back(layer); net.shrink_to_fit();

            actlist.push_back(list (neuron_count, 0)); actlist.shrink_to_fit();

            lastlayer++;

            initialised = true;
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
            if (input.size() != incount) throw runtime_error("Input list should be the same size as the amount of input neurons.");
            
            in = input;

            for (int i = 0; i < net.size() - 1; i++){
                for (int j = 0; j < net[i].size(); j++){
                    if (i == 0) actlist[i][j] = calc_act(in, &net[i][j]);
                    else actlist[i][j] = calc_act(actlist[i-1], &net[i][j]);
                }
            }

            for (int i = 0; i < net[lastlayer].size(); i++){
                if (net.size() == 1) actlist[lastlayer][i] = calc_act(in, &net[lastlayer][i]);
                else actlist[lastlayer][i] = calc_act(actlist[net.size()-1], &net[lastlayer][i]);
            }
            
            type temp = actlist[lastlayer][0],
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
    
        type loss (list wanted, loss_func function = MEAN_SQUARED){ // calculates the cost, a measure of how bad the net is performing (only works if neural_network::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_network::add_dense_layer to initialise the net.");
            int size = actlist[lastlayer].size();
            if (wanted.size() != size) throw runtime_error("Wanted list should be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_network::evaluate has already been run.");

            type cost = 0;
            for (int i = 0; i < size; i++) cost += lfunc(function)(wanted[i], actlist[lastlayer][i], 0);

            return cost;
        }

        void fit (list wanted, type learning_rate, int batch_size = 1, loss_func function = MEAN_SQUARED){ // improves the net by trying to lower the cost (only works if neural_network::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run neural_network::add_dense_layer to initialise the net.");
            list impact_list;
            int size = net.size(),
            outsize = net[lastlayer].size();

            if (wanted.size() != outsize) throw runtime_error("Wanted list should be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if neural_network::evaluate has already been run.");
            
            for (int i = 0; i < outsize; i++){
                neuron neur = net[lastlayer][i];
                int wgtsize = neur.wgt.size();

                if (size == 0) impact_list = calc_impact(in, &neur, wanted[i], actlist[lastlayer][i], function, learn_coeffficents);
                else impact_list = calc_impact(actlist[size-1], &neur, wanted[i], actlist[lastlayer][i], function, learn_coeffficents);

                neur.bias -= (learning_rate * impact_list[0] / batch_size);

                for (int j = 0; j < wgtsize; j++){
                    neur.wgt[j] -= (learning_rate * impact_list[j+1] / batch_size);
                    if (size != 1) change_previous(size-1, impact_list[j+wgtsize+1], learning_rate, batch_size);
                }
            }
        }
};


class CNN : public neural_network {
    private:
        bool fullyconnected = false; // boolean for storing id the user already added fullyconnected layers

        // used for readability
        const int CONVDETAILS = 7, // amount of details in a convolutional layer
            POOLDETAILS = 5, // amount of details in a pooling layer
            TYPESPECIFIER = 0;
        
        enum convdetail{dummy,FEATURES,XPADDING,YPADDING,XFEATURE,YFEATURE,STRIDEFEATURE,CONVOLUTIONINDEX};
        enum pooldetails{dummy,MAXPOOL,XPOOLWINDOW,YPOOLWINDOW,STRIDEPOOLWINDOW};

        enum layertype{CONVOLUTIONAL,POOLING};


        void printconv(bool term, bool file){

        }

        type pool (matrix in, bool m){
            int rows = in[0].size(), col = in.size();
            type returnval = 0;

            for (int i = 0; i < col; i++) for (int j = 0; j < rows; j++){
                if (in[i][j] > returnval && m) returnval = in[i][j];
                if (!m) returnval += in[i][j];
            }

            return m ? returnval : returnval / (rows * col);
        }

        matrix backpool (matrix in, type out, bool m){
            int rows = in[0].size(), col = in.size();
            matrix returnval;
            if (m) returnval = matrix(col, list (rows, 0));
            else {returnval = matrix(col, list (rows, 1/(rows*col))); return returnval;}

            for (int i = 0; i < col; i++) for (int j = 0; j < rows; j++)
                if (m && in[i][j] == out) {returnval[i][j] = 1; break;}
            

            return returnval;
        }

        type conv(matrix I, matrix F){
            type returnval = 0;
            int rows = I[0].size();
            
            for (int i = 0; i < I.size(); i++) for (int j = 0; j < rows; j++) returnval += I[i][j] * F[i][j];

            return ReLU(returnval, 0, 0);
        }

    public:
        vector<vector<matrix>> K, // fourth order tensor for storing all the features
        channels, // fourth order tensor for storing all the convolutional channels
        backprop; // fourth order tensor for storing all the partial derivatives
        matrix biasses, // matrix for storing biasses of convolution layers

        conlayers;

        int xin, yin, inANN,
        outchan,
        lastconlayer = -1;

        void input (int xinput, int yinput, int channelcount = 1){
            if (xinput <= 0 || yinput <= 0 || channelcount <=0) throw runtime_error("All parameters of CNN::input should be greater than zero.");
            
            xin = xinput; yin = yinput; outchan = channelcount;

            // adding the input matrix to CNN::channels
            channels.push_back(vector<matrix> (channelcount, matrix (yin, list (xin))));
        }

        void add_convolutional_layer (int features, bool paddingvalid, int xfeature = 3, int yfeature = 3, int stridefeature = 1){
            if (fullyconnected) throw runtime_error("No convolutional layers should be added after adding a fully connected layer.");
            if (conlayers.size() == 0){
                
            }
            
            int xpa, ypa;
            if (paddingvalid){xpa = (xfeature-1)/2; ypa = (yfeature-1)/2;}
            else {xpa = 0; ypa = 0;}

            list layer (CONVDETAILS);
            layer[TYPESPECIFIER] = CONVOLUTIONAL;
            layer[FEATURES] = features;
            layer[XPADDING] = xpa;
            layer[YPADDING] = ypa;
            layer[XFEATURE] = xfeature;
            layer[YFEATURE] = yfeature;
            layer[STRIDEFEATURE] = stridefeature;
            layer[CONVOLUTIONINDEX] = K.size();

            conlayers.push_back(layer); conlayers.shrink_to_fit();
            lastconlayer++;
            K.push_back(vector<matrix> (features, matrix (yfeature, list(xfeature, 1)))); K.shrink_to_fit();
            biasses.push_back(list (features, 0)); biasses.shrink_to_fit();
            
            // calculating the new size of the vector of matrices that will be pushed to CNN::channels
            outchan *= features;
            int newY = floor(1 + ((channels[conlayers.size()-2][0].size()+2*ypa - yfeature)/stridefeature)),
                newX = floor(1 + ((channels[conlayers.size()-2][0][0].size()+2*xpa - xfeature)/stridefeature));
            channels.push_back(vector<matrix> (outchan, matrix (newY, list(newX)))); channels.shrink_to_fit();

            inANN = outchan*newX*newY;
        }

        void add_pooling_layer (bool maxpooling, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int xinput = 0, int yinput = 0){
            if (fullyconnected) throw runtime_error("No pooling layers should be added after adding a fully connected layer.");
            if (conlayers.size() == 0){
                if (xinput == 0 || yinput == 0) throw runtime_error("For the first layer, xinput and yinput should be defined as a number greater than 0.");
                xin = xinput; yin = yinput;

                // adding the input matrix to CNN::channels
                channels.push_back(vector<matrix> (1, matrix (yin, list (xin))));
            }
            
            list layer (POOLDETAILS);
            layer[TYPESPECIFIER] = POOLING;
            layer[MAXPOOL] = maxpooling;
            layer[XPOOLWINDOW] = xpoolwindow;
            layer[YPOOLWINDOW] = ypoolwindow;
            layer[STRIDEPOOLWINDOW] = stridepoolwindow;

            conlayers.push_back(layer); conlayers.shrink_to_fit();
            lastconlayer++;

            int newY = floor(1 + ((channels[conlayers.size()-2][0].size() - ypoolwindow)/stridepoolwindow)),
                newX = floor(1 + ((channels[conlayers.size()-2][0][0].size() - xpoolwindow)/stridepoolwindow));
            channels.push_back(vector<matrix> (outchan, matrix (newY, list(newX, 0)))); channels.shrink_to_fit();

            inANN = outchan*newX*newY;
        }

        void add_dense_layer (int neuron_count, act_func activation_function, type coefficient = 0.01){
            if (conlayers.size() == 0) throw runtime_error("Please add convolutional or pooling layers first.");
            fullyconnected = true;

            neural_network::input(inANN);
            neural_network::add_dense_layer(neuron_count, activation_function, coefficient);
        }

        void export_net (bool write_to_terminal = true, bool write_to_file = false){ // method for printing the CNN
            if (!neural_network::initialised) throw runtime_error("Please run CNN::add_convolutional_layer or CNN::add_pooling_layer and then neural_network::add_dense_layer to initialise the CNN.");
            printconv(write_to_terminal, write_to_file); neural_network::export_net(write_to_terminal, write_to_file);
        }

        int evaluate (vector<matrix> input){
            if (!neural_network::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            
            for (int c = 0; c < input.size(); c++) // looping over the vector of input channels, for example RGBA channels
                channels[0][c] = input[c];

            // convolution and pooling layers
            matrix I, temp;

            for (int lay = 0; lay < lastconlayer+1; lay++){ // lastconlayer is an index, +1 makes us iterate over all layers
                list layer = conlayers[lay];

                int ysize = channels[lay+1][0].size(), // first element in conlayers is the input matrix, +1 starts us of at the first convolutional layer
                    xsize = channels[lay+1][0][0].size(),
                    ypa = 0, xpa = 0;
                
                if (layer[TYPESPECIFIER] == CONVOLUTIONAL){
                    ypa = layer[YPADDING]; xpa = layer[XPADDING];

                    I = matrix (layer[YFEATURE], list (layer[XFEATURE]));
                    temp = matrix (ysize + 2*ypa, list (xsize + 2*xpa, 0));
                } else I = matrix (layer[YPOOLWINDOW], list (layer[XPOOLWINDOW]));

                for (int chan = 0; chan < channels[lay+1].size(); chan++){
                    int feature, prevchannel = chan;
                    if (layer[TYPESPECIFIER] == CONVOLUTIONAL){
                        int f = layer[FEATURES];
                        feature = floor(chan / f);
                        prevchannel = chan % f;
                    }

                    for (int y = 0; y < ysize; y++) for (int x = 0; x < xsize; x++){
                        if (layer[TYPESPECIFIER] == CONVOLUTIONAL){ // convolution
                            insmatrix(temp, channels[lay][prevchannel], ypa, xpa);

                            for (int yI = 0; yI < layer[YFEATURE]; yI++) for (int xI = 0; xI < layer[XFEATURE]; xI++)
                                I[yI][xI] = temp[yI + y*layer[STRIDEFEATURE]][xI + x*layer[STRIDEFEATURE]];

                            channels[lay+1][chan][y][x] = conv(I, K[layer[CONVOLUTIONINDEX]][feature]);
                        } else { // pooling
                            for (int yI = 0; yI < layer[YPOOLWINDOW]; yI++) for (int xI = 0; xI < layer[XPOOLWINDOW]; xI++)
                                I[yI][xI] = channels[lay][prevchannel][yI + y*layer[STRIDEPOOLWINDOW]][xI + x*layer[STRIDEPOOLWINDOW]];
                            
                            channels[lay+1][chan][y][x] = pool(I, layer[MAXPOOL]);
                        }
                    }
                }
            }

            // flattening
            list flattened (inANN);
            for (int i = 0; i < outchan; i++) for (int j = 0; j < channels[lastconlayer+1][0].size(); j++) for (int k = 0; k < channels[lastconlayer][0][0].size(); k++)
                flattened.push_back(channels[lastconlayer+1][i][j][k]);
            flattened.shrink_to_fit();

            return neural_network::evaluate(flattened); // feedforward layers
        }

        void fit (list wanted, type learning_rate, int batch_size = 1){
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