# include <fstream>
# include <iostream>

# include "neuron.cpp"

using std::runtime_error,
    std::cout,
    std::ofstream;

class ANN {
    protected: // used in inherited classes
        list dcdin;
        bool initialised = false,
        inputinitialised = false;
    private:
        matrix dcda;

        void calc_previous (int layer, int index, type da){ // recursive method for changing the previous activation
            dcda[layer-1][index] *= da; // minus one, because dcda does not contai the last layer of the net
            if (layer){
                for (int w = 0; w < net[layer][index].wgt.size(); w++) calc_previous(layer, w, net[layer][index].wgt[w]);
            }
        }
    public:
        // variables for middle and output layers of the net
        Net net;
        matrix actlist;
        
        list in, // variable for storing the given input of the net
        outlist; // variable for storing the output list of the net (the same as actlist[lastlayer])

        type certainty; // variable for storing the certainty of the net

        int out_index = 0, // variable for storing the index of the highest value in ANN::outlist
        lastlayer = -1, // variable for storing the index of the last layer
        incount;

        unsigned long long iteration = 0; // variable for storing the number of feed forward cycles the net has gone through        

        void add_input (int input_count){
            if (input_count <= 0) throw runtime_error("input_count should be greater than zero.");
            if (inputinitialised) throw runtime_error("Input was already defined.");
    	    
            incount = input_count;

            inputinitialised = true;
        }

        void add_dense_layer (int neuron_count, act_func activation_function, type coefficient = 0.01){
            if (!inputinitialised) throw runtime_error("ANN::add_input should have been run before adding dense layers.");

            Collumn layer (neuron_count);
            int wgtsize;
            for (int i = 0; i < neuron_count; i++){
                if (net.size() == 0) wgtsize = incount;
                else wgtsize = net[lastlayer].size();

                layer[i].wgt = list (wgtsize, 1);
                layer[i].coef = coefficient;

                if (activation_function != SOFTMAX) layer[i].func = func(activation_function);
                else layer[i].softmax = true;
            }
            net.push_back(layer); net.shrink_to_fit();
            actlist.push_back(list (neuron_count, 0)); actlist.shrink_to_fit();

            if (lastlayer >= 0) dcda.push_back(list (net[lastlayer].size(), 1)); dcda.shrink_to_fit();

            lastlayer++;

            initialised = true;
        }

        void export_net (bool write_to_terminal = true, bool write_to_file = false, const char * path = "data.txt"){ // method for printing the net
            if (!initialised) throw runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");

            if (!write_to_file && !write_to_terminal) cout << "ok...";

            ofstream outputFile;
            if (write_to_file) outputFile = ofstream (path); // write data to a new file called "data.txt"

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
            if (!initialised) throw runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");
            if (input.size() != incount) throw runtime_error("Input list should be the same size as the amount of input neurons.");
            
            in = input;

            for (int i = 0; i < net.size() - 1; i++){
                for (int j = 0; j < net[i].size(); j++){
                    if (i == 0) actlist[i][j] = calc_act(in, &net[i][j]);
                    else actlist[i][j] = calc_act(actlist[i-1], &net[i][j]);
                }
                if (net[i][0].softmax) actlist[i] = softmax(actlist[i], net[i][0].coef);
            }

            for (int i = 0; i < net[lastlayer].size(); i++){
                if (net.size() == 1) actlist[lastlayer][i] = calc_act(in, &net[lastlayer][i]);
                else actlist[lastlayer][i] = calc_act(actlist[net.size()-1], &net[lastlayer][i]);
            }
            if (net[lastlayer][0].softmax) actlist[lastlayer] = softmax(actlist[lastlayer], net[lastlayer][0].coef);
            
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
    
        type loss (list wanted, loss_func function = MEAN_SQUARED){ // calculates the cost, a measure of how bad the net is performing (only works if ANN::calc_out has already been run)
            if (!initialised) throw runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");
            int size = actlist[lastlayer].size();
            if (wanted.size() != size) throw runtime_error("Wanted list should be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if ANN::evaluate has already been run.");

            type cost = 0;
            for (int i = 0; i < size; i++) cost += lfunc(function)(wanted[i], actlist[lastlayer][i], 0);

            return cost;
        }

        void fit (list wanted, type learning_rate, int batch_size = 1, loss_func function = MEAN_SQUARED, bool learn_coeffficents = false){ // improves the net by trying to lower the cost (only works if ANN::calc_out has already been run)
            int outsize = net[lastlayer].size();
            list impact_list;

            if (wanted.size() != outsize) throw runtime_error("Wanted list should be the same size as the number of outputs");
            if (iteration == 0) throw runtime_error("This method only works if ANN::evaluate has already been run.");
            if (!initialised) throw runtime_error("Please run ANN::input and ANN::add_dense_layer to initialise the net.");

            if (lastlayer) list (net[lastlayer-1].size(), 1).swap(dcda[lastlayer-1]);

            for (int neur = 0; neur < outsize; neur++){ // looping over output neurons to change their parameters and calculate the derivative with respect to the activation of previous layer
                int wgtsize = net[lastlayer][neur].wgt.size();

                if (lastlayer == 0) impact_list = calc_impact(in, &net[lastlayer][neur], wanted[neur], actlist[lastlayer][neur], function, learn_coeffficents);
                else impact_list = calc_impact(actlist[lastlayer], &net[lastlayer][neur], wanted[neur], actlist[lastlayer][neur], function, learn_coeffficents);

                if (learn_coeffficents) net[lastlayer][neur].coef -= learning_rate * impact_list[0] / batch_size;
                net[lastlayer][neur].bias -= learning_rate * impact_list[learn_coeffficents] / batch_size;

                for (int w = 0; w < wgtsize; w++){
                    if (lastlayer) dcda[lastlayer-1][w] *= net[lastlayer][neur].wgt[w];
                    net[lastlayer][neur].wgt[w] -= (learning_rate * impact_list[w+1+learn_coeffficents] / batch_size);
                }
            }

            // calculating the rest of the derivatives
            for (int bblay = lastlayer-2; bblay >= -1; bblay++){ // looping through blackbox layers and input layer
                int ncount;
                list * impactlayer;

                if (bblay != -1) impactlayer = &dcda[bblay];
                else impactlayer = &dcdin;

                ncount = ( * impactlayer ).size();
                list (ncount, 1).swap( * impactlayer );
                
                for (int neur = 0; net[bblay+1].size(); neur++) for (int w = 0; w < ncount; w++)
                    ( * impactlayer )[w] *= net[bblay+1][neur].wgt[w];
            }

            for (int layer = 0; layer < lastlayer; layer++) // changing the middle layers
                for (int neur = 0; neur < net[layer].size(); layer++){
                    list impact_list;
                    int wgtsize = net[layer][neur].wgt.size();

                    double constant = learning_rate / batch_size;

                    if (layer == 0) impact_list = calc_impact(in, &net[layer][neur], actlist[lastlayer][neur], learn_coeffficents);
                    else impact_list = calc_impact(actlist[layer-1], &net[layer][neur], actlist[lastlayer][neur], learn_coeffficents);

                    if (learn_coeffficents) net[layer][neur].coef -= constant * dcda[layer][neur] * impact_list[0];
                    net[layer][neur].bias -= constant * dcda[layer][neur] * impact_list[learn_coeffficents];


                    for (int j = 0; j < wgtsize; j++)
                        net[layer][neur].wgt[j] -= constant * dcda[layer][neur] * impact_list[j+1+learn_coeffficents];
                }
            
        }
};


class CNN : public ANN {
    private:
        bool fullyconnected = false; // boolean for storing id the user already added fullyconnected layers

        // used for readability
        enum convdetail{FEATURES,XPADDING,YPADDING,XFEATURE,YFEATURE,STRIDEFEATURE,CONVOLUTIONINDEX};
        enum pooldetails{MAXPOOL,XPOOLWINDOW,YPOOLWINDOW,STRIDEPOOLWINDOW};

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
                if (in[i][j] == out) {returnval[i][j] = 1; break;}
            

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
        matrix biasses; // matrix for storing biasses of convolution layers

        vector<layerdetails> conlayers;

        int xin, yin,
        inANN, xinANN, yinANN,
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

            layerdetails layer;
            layer.type = CONVOLUTIONAL;
            layer.convolutional[FEATURES] = features;
            layer.convolutional[XPADDING] = xpa;
            layer.convolutional[YPADDING] = ypa;
            layer.convolutional[XFEATURE] = xfeature;
            layer.convolutional[YFEATURE] = yfeature;
            layer.convolutional[STRIDEFEATURE] = stridefeature;
            layer.convolutional[CONVOLUTIONINDEX] = K.size();

            conlayers.push_back(layer); conlayers.shrink_to_fit();
            lastconlayer++;
            K.push_back(vector<matrix> (features, matrix (yfeature, list(xfeature, 1)))); K.shrink_to_fit();
            biasses.push_back(list (features, 0)); biasses.shrink_to_fit();
            
            // calculating the new size of the vector of matrices that will be pushed to CNN::channels
            outchan *= features;
            int newY = floor(1 + ((channels[conlayers.size()-2][0].size()+2*ypa - yfeature)/stridefeature)),
                newX = floor(1 + ((channels[conlayers.size()-2][0][0].size()+2*xpa - xfeature)/stridefeature));
            channels.push_back(vector<matrix> (outchan, matrix (newY, list(newX)))); channels.shrink_to_fit();
            backprop.push_back(vector<matrix> (outchan, matrix (newY, list(newX)))); backprop.shrink_to_fit();

            inANN = outchan*newX*newY;
            xinANN = newX;
            yinANN = newY;
        }

        void add_pooling_layer (bool maxpooling, int xpoolwindow = 2, int ypoolwindow = 2, int stridepoolwindow = 2, int xinput = 0, int yinput = 0){
            if (fullyconnected) throw runtime_error("No pooling layers should be added after adding a fully connected layer.");
            if (conlayers.size() == 0){
                if (xinput == 0 || yinput == 0) throw runtime_error("For the first layer, xinput and yinput should be defined as a number greater than 0.");
                xin = xinput; yin = yinput;

                // adding the input matrix to CNN::channels
                channels.push_back(vector<matrix> (1, matrix (yin, list (xin))));
            }
            
            layerdetails layer;
            layer.type = POOLING;
            layer.pooling[MAXPOOL] = maxpooling;
            layer.pooling[XPOOLWINDOW] = xpoolwindow;
            layer.pooling[YPOOLWINDOW] = ypoolwindow;
            layer.pooling[STRIDEPOOLWINDOW] = stridepoolwindow;

            conlayers.push_back(layer); conlayers.shrink_to_fit();
            lastconlayer++;

            int newY = floor(1 + ((channels[conlayers.size()-2][0].size() - ypoolwindow)/stridepoolwindow)),
                newX = floor(1 + ((channels[conlayers.size()-2][0][0].size() - xpoolwindow)/stridepoolwindow));
            channels.push_back(vector<matrix> (outchan, matrix (newY, list(newX, 0)))); channels.shrink_to_fit();

            inANN = outchan*newX*newY;
            xinANN = newX;
            yinANN = newY;
        }

        void add_dense_layer (int neuron_count, act_func activation_function, type coefficient = 0.01){
            if (conlayers.size() == 0) throw runtime_error("Please add convolutional or pooling layers first.");
            fullyconnected = true;

            ANN::add_input(inANN);
            ANN::add_dense_layer(neuron_count, activation_function, coefficient);
        }

        void export_net (bool write_to_terminal = true, bool write_to_file = false){ // method for printing the CNN
            if (!ANN::initialised) throw runtime_error("Please run CNN::add_convolutional_layer or CNN::add_pooling_layer and then ANN::add_dense_layer to initialise the CNN.");
            printconv(write_to_terminal, write_to_file); ANN::export_net(write_to_terminal, write_to_file);
        }

        int evaluate (vector<matrix> input){
            if (!ANN::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");
            
            for (int c = 0; c < input.size(); c++) // looping over the vector of input channels, for example RGBA channels
                channels[0][c] = input[c];

            // convolution and pooling layers
            matrix I, temp;

            for (int lay = 0; lay < lastconlayer+1; lay++){ // lastconlayer is an index, +1 makes us iterate over all layers
                layerdetails layer = conlayers[lay];

                int ysize = channels[lay+1][0].size(), // first element in conlayers is the input matrix, +1 starts us of at the first convolutional layer
                    xsize = channels[lay+1][0][0].size(),
                    ypa = 0, xpa = 0;
                
                if (layer.type == CONVOLUTIONAL){
                    ypa = layer.convolutional[YPADDING]; xpa = layer.convolutional[XPADDING];

                    I = matrix (layer.convolutional[YFEATURE], list (layer.convolutional[XFEATURE]));
                    temp = matrix (ysize + 2*ypa, list (xsize + 2*xpa, 0));
                } else I = matrix (layer.pooling[YPOOLWINDOW], list (layer.pooling[XPOOLWINDOW]));

                for (int chan = 0; chan < channels[lay+1].size(); chan++){
                    int feature, prevchannel = chan;
                    if (layer.type == CONVOLUTIONAL){
                        int f = layer.convolutional[FEATURES];
                        feature = floor(chan / f);
                        prevchannel = chan % f;
                    }

                    for (int y = 0; y < ysize; y++) for (int x = 0; x < xsize; x++){
                        if (layer.type == CONVOLUTIONAL){ // convolution
                            insmatrix(temp, channels[lay][prevchannel], ypa, xpa);

                            for (int yI = 0; yI < layer.convolutional[YFEATURE]; yI++) for (int xI = 0; xI < layer.convolutional[XFEATURE]; xI++)
                                I[yI][xI] = temp[yI + y*layer.convolutional[STRIDEFEATURE]][xI + x*layer.convolutional[STRIDEFEATURE]];

                            channels[lay+1][chan][y][x] = conv(I, K[layer.convolutional[CONVOLUTIONINDEX]][feature]) + biasses[lay][feature];
                        } else { // pooling
                            for (int yI = 0; yI < layer.pooling[YPOOLWINDOW]; yI++) for (int xI = 0; xI < layer.pooling[XPOOLWINDOW]; xI++)
                                I[yI][xI] = channels[lay][prevchannel][yI + y*layer.pooling[STRIDEPOOLWINDOW]][xI + x*layer.pooling[STRIDEPOOLWINDOW]];
                            
                            channels[lay+1][chan][y][x] = pool(I, layer.pooling[MAXPOOL]);
                        }
                    }
                }
            }

            // flattening
            list flattened (inANN);
            for (int i = 0; i < outchan; i++) for (int j = 0; j < channels[lastconlayer+1][0].size(); j++) for (int k = 0; k < channels[lastconlayer][0][0].size(); k++)
                flattened.push_back(channels[lastconlayer+1][i][j][k]);
            flattened.shrink_to_fit();

            return ANN::evaluate(flattened); // feedforward layers
        }

        void fit (list wanted, type learning_rate, int batch_size = 1){
            if (!ANN::initialised) throw runtime_error("Please run CNN::init or CNN::modify to initialise the CNN.");

            ANN::fit(wanted, learning_rate); // first step of backprop: the fully connected layers

            // second step: passing on del cost by del input list
            int index = 0;
            for (int i = 0; i < outchan; i++) for (int j = 0; j < yinANN; j++) for (int k = 0; k < xinANN; k++){
                backprop[lastconlayer][i][j][k] = dcdin[index];
                index++;
            }

            // last step: propagating backwards through convolutional layers
            for (int lay = lastconlayer; lay > 0; lay--){ // propagating backwards through convolutional layers
                layerdetails layer = conlayers[lay];
                for (int c = 0; c < backprop[lay].size(); c++)
                    for (int y = 0; y < backprop[lay][c].size(); y++)
                        for (int x = 0; x < backprop[lay][c][y].size(); x++){
                            if (layer.type == CONVOLUTIONAL){ // convolution

                            } else { // pooling
                                
                            }
                        }
            }
        }
};

class RNN : public ANN {
    private:
    public:
        void add_input (int input_count){
            ANN::add_input(2*input_count);
        }

        list evaluate (list input){
            
        }
};

class LSTM : public ANN {
    
};