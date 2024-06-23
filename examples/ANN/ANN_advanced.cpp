# include <iostream>
# include <fstream>

// # define FLOAT
# include "..\..\neural_net.cpp"

using std::ostream,
    std::ifstream,
    std::endl, std::cout;

const int epochs = 5,
    batch_size = 10;

const double initial_lr = .5,
    decay = 1;

int trainsize, testsize,
    iterations;
vector<matrix> traindata, testdata;
vector<unsigned char> trainlbl, testlbl;

void writePixel (int pixel){ // This function is for writing a pixel value to the screen. When it writes certain characters was randomly chosen
    switch (pixel){
        case 0:
            cout << "  ";
            break;
        case 1 ... 50:
            cout << ". ";
            break;
        case 51 ... 100:
            cout << "+ ";
            break;
        case 101 ... 150:
            cout << "Z ";
            break;
        case 151 ... 200:
            cout << "B ";
            break;
        case 201 ... 255:
            cout << "@ ";
            break;
        default:
            break;
    }
}
void writeImage (matrix image){
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++) writePixel(image[i][j]);
        cout << endl;
    }
}

// The following two functions are to read the MNIST dataset of labeled,  handwritten digits. Credit: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
int reverseInt (int i){
    unsigned char c1,  c2,  c3,  c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist (const char * path, const char * lblpath, bool test){
    ifstream file (path, std::ios::binary);
    ifstream lblfile (lblpath, std::ios::binary);

    if (file.is_open() && lblfile.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        int magic_number2 = 0;
        int number_of_labels = 0;

        file.read(( char * ) &magic_number, sizeof(magic_number)); 
        magic_number = reverseInt(magic_number);
        file.read(( char * ) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read(( char * ) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read(( char * ) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        lblfile.read((char *)&magic_number2, sizeof(magic_number2));
        magic_number2 = reverseInt(magic_number2);
        lblfile.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        if (!test){
            traindata = vector<matrix> (number_of_images, matrix (n_rows, list (n_cols)));
            trainlbl = vector<unsigned char> (number_of_images);
            iterations = number_of_images;
        } else {
            testdata = vector<matrix> (number_of_images, matrix (n_rows, list (n_cols)));
            testlbl = vector<unsigned char> (number_of_images);
        }

        for (int i = 0; i < number_of_images; i++){
            for (int r = 0; r < n_rows; r++){
                for (int c = 0; c < n_cols; c++){
                    unsigned char temp = 0;
                    file.read(( char * ) &temp, sizeof(temp));
                    traindata[i][r][c] = temp;
                }
            }
            unsigned char temp = 0;
            lblfile.read(( char* ) &temp, sizeof(temp));
        }
    } else {
        cout << "file did not open" << endl;
    }

    file.close();
    lblfile.close();
}

// Template to print a vector using cout. Credit: https://www.geeksforgeeks.org/different-ways-to-print-elements-of-vector/
template <typename S> ostream& operator<<(ostream& os, const vector<S>& vector){
    for (auto element : vector) os << element << " ";
    return os;
}

list generate_wanted (unsigned char wanted){
    list retval (10, 0);
    retval[wanted] = 1;
    return retval;
}

double lr (int epoch){
    return initial_lr / (1 + decay*epoch);
}

void test (int index, ANN * net){
    writeImage(traindata[index]);
    cout << "actual answer: " << trainlbl[index] << "\nmodel prediction: " << net->eval(flatten(traindata[index])) << " certainty: " << net->certainty << "%";
}

int main (){
    read_mnist( // traindata
        "C:\\Users\\OE104296\\OneDrive - Libreon\\Documenten\\GitHub\\neural_networking\\examples\\MNIST\\t10k-images.idx3-ubyte",
        "C:\\Users\\OE104296\\OneDrive - Libreon\\Documenten\\GitHub\\neural_networking\\examples\\MNIST\\t10k-labels.idx1-ubyte",
        false
    );
    cout << "done loading training data\n";
    // read_mnist( // testdata
    //     "C:\\Users\\OE104296\\OneDrive - Libreon\\Documenten\\GitHub\\neural_networking\\examples\\MNIST\\train-images.idx3-ubyte",
    //     "C:\\Users\\OE104296\\OneDrive - Libreon\\Documenten\\GitHub\\neural_networking\\examples\\MNIST\\train-labels.idx1-ubyte",
    //     true
    // );

    ANN net;
        net.add_input(28*28);
        net.add_dense_layer(50, SILU, 1);
        net.add_dense_layer(10, SOFTMAX, 1);

    double cost = 0;
    for (int epoch = 0; epoch < epochs; epoch++){
        for (int iteration = 0; iteration < iterations; iteration++){
            list wanted = generate_wanted(trainlbl[iteration]);

            net.eval(flatten(traindata[iteration]));
            net.fit(wanted, lr(epoch), batch_size, CROSS_ENTROPY);

            cost += net.loss(wanted, CROSS_ENTROPY);
        }
        cout << "cost:" << cost/iterations << "\n";
        cost = 0;
    }

    test(0, &net);
    test(1, &net);
    test(2, &net);


    return 0;
}