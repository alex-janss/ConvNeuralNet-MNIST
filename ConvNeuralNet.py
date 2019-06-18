import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.signal import convolve2d


np.set_printoptions(linewidth=300)
image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "C:/Users/Alex/PycharmProjects/ConvNeuralNet/"
# saves train and test data to pickle file
# makes loading faster for future runs
try:
    train_data = pickle.load(open("train_data.pickle", "rb"))
    test_data = pickle.load(open("test_data.pickle", "rb"))
except (OSError, IOError) as e:
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
    pickle.dump(train_data, open("train_data.pickle", "wb"))
    pickle.dump(test_data, open("test_data.pickle", "wb"))

fac = .99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + .01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + .01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels = (lr == train_labels).astype(np.float)
test_labels = (lr == test_labels).astype(np.float)


def sigmoid(input_vector):
    return np.arctan(input_vector) / np.pi + .5


def deriv_sigmoid(input_vector):
    return 1. / (np.pi * (1 + input_vector ** 2))


def max_pool(A, nrow = 2, ncol = 2):
    return A.reshape(A.shape[0] // nrow, nrow, A.shape[1] // ncol, ncol).max(axis=(1,3))


def inverse_pool(A, nrow = 2, ncol = 2):
    return np.kron(A, np.ones((nrow, ncol)) * 1/(nrow*ncol))


def pad(A, width):
    Z = np.zeros((A.shape[0]+2*width,A.shape[1]+2*width))
    Z[width:A.shape[0]+width,width:A.shape[1]+width] = A
    return Z

# shuffles a pair of vectors together,
# i.e. both vectors are shuffled exactly the same way
def shuffle_together(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

class CNN:

    def __init__(self, num_filters=1):
        self.num_filters = num_filters
        self.input_layer = np.zeros((28,28))
        self.padded_input = np.zeros((32,32))
        self.filtered_layer1 = np.zeros((28,28))
        self.filtered_z1 = np.zeros((28,28))
        self.pooled_layer1 = np.zeros((14,14))
        self.pooled_z1 = np.zeros((14,14))
        self.padded_pool1 = np.zeros((18,18))
        self.filtered_layer2 = np.zeros((14,14))
        self.filtered_z2 = np.zeros((14, 14))
        self.pooled_layer2 = np.zeros((7, 7))
        self.pooled_z2 = np.zeros((7, 7))
        self.output_layer = np.zeros((10,1))
        self.output_z = np.zeros((10,1))

        self.filter1 = np.zeros((5,5))
        self.filter_bias1 = np.zeros(self.num_filters)
        self.filter2 = np.zeros((5,5))
        self.filter_bias2 = np.zeros(self.num_filters)
        self.weights = np.zeros((10,self.pooled_layer2.size))
        self.biases = np.zeros((10,1))
        self.errors = np.zeros((10,1))


    def random_init(self):
        self.filter1 = np.random.standard_normal(self.filter1.shape)
        self.filter_bias1 = np.random.standard_normal(self.filter_bias1.shape)
        self.filter2 = np.random.standard_normal(self.filter2.shape)
        self.filter_bias2 = np.random.standard_normal(self.filter_bias2.shape)
        self.weights = np.random.standard_normal(self.weights.shape)
        self.biases = np.random.standard_normal(self.biases.shape)

    def get_cost(self):
        return la.norm(self.errors)

    def add_nets(self, input_net):
        self.filter1 += input_net.filter1
        self.filter_bias1 += input_net.filter_bias1
        self.filter2 += input_net.filter2
        self.filter_bias2 += input_net.filter_bias2
        self.weights += input_net.weights
        self.biases += input_net.biases

    def scale_net(self, factor):
        self.filter1 *= factor
        self.filter_bias1 *= factor
        self.filter2 *= factor
        self.filter_bias2 *= factor
        self.weights *= factor
        self.biases *= factor
        return self


    def classify(self, input_img, input_label):
        # input layer
        self.input_layer = np.reshape(input_img, self.input_layer.shape)
        self.padded_input = pad(self.input_layer, 2)
        # filtered layer 1
        self.filtered_z1 = correlate2d(self.padded_input,self.filter1,mode="valid") + self.filter_bias1
        self.filtered_layer1 = sigmoid(self.filtered_z1)
        # pooling 1
        self.pooled_z1 = max_pool(self.filtered_z1)
        self.pooled_layer1 = sigmoid(self.pooled_z1)
        self.padded_pool1 = pad(self.pooled_layer1, 2)
        # filtered layer 2
        self.filtered_z2 = correlate2d(self.padded_pool1,self.filter2,mode="valid") + self.filter_bias2
        self.filtered_layer2 = sigmoid(self.filtered_z2)
        # pooling 2
        self.pooled_z2 = max_pool(self.filtered_z2)
        self.pooled_layer2 = sigmoid(self.pooled_z2)

        # output layer
        self.output_z = (self.weights @ self.pooled_layer2.reshape(self.pooled_layer2.size,1)) + self.biases
        self.output_layer = sigmoid(self.output_z)
        self.errors = self.output_layer - input_label.reshape(input_label.size,1)


    def backprop(self):
        '''
        grad_net = CNN(self.num_filters)
        # errors L3
        grad_net.output_layer = 2 * self.errors * deriv_sigmoid(self.output_z)
        # biases L3
        grad_net.biases = grad_net.output_layer
        # weights L3
        grad_net.weights = np.outer(grad_net.output_layer, self.pooled_layer.reshape(self.pooled_layer.size))
        # errors L2
        grad_net.pooled_layer = np.reshape((self.weights.T @ grad_net.output_layer), self.pooled_layer.shape) * deriv_sigmoid(self.pooled_z)
        # errors L1
        for i in range(self.num_filters):
            grad_net.filtered_layer[i] = inverse_pool(grad_net.pooled_layer[i])
        # bias L1
        grad_net.filter_bias = np.sum(grad_net.filtered_layer, axis=(1,2))
        # filter weights L1
        for i in range(self.num_filters):
            grad_net.filter[i] = correlate2d(self.padded_input, grad_net.filtered_layer[i], mode="valid")

         '''
        grad_net = CNN(self.num_filters)
        # error L5
        grad_net.output_layer = 2 * self.errors * deriv_sigmoid(self.output_z)
        # errors L4
        grad_net.pooled_layer2 = np.reshape((self.weights.T @ grad_net.output_layer), self.pooled_layer2.shape) * deriv_sigmoid(self.pooled_z2)
        # errors L3
        grad_net.filtered_layer2 = inverse_pool(grad_net.pooled_layer2)
        # errors L2
        grad_net.pooled_layer1 = convolve2d(pad(self.filter2, 11), grad_net.filtered_layer2, mode='valid') * deriv_sigmoid(self.pooled_z1)
        # errors L1
        grad_net.filtered_layer1 = inverse_pool(grad_net.pooled_layer1)

        # weights
        # weights L5
        grad_net.weights = np.outer(grad_net.output_layer, self.pooled_layer2.reshape(self.pooled_layer2.size))
        # weights L3
        grad_net.filter2 = correlate2d(self.padded_pool1, grad_net.filtered_layer2, mode='valid')
        # weights L1
        grad_net.filter1 = correlate2d(self.padded_input, grad_net.filtered_layer1, mode='valid')

        #biases
        # biases L5
        grad_net.biases = grad_net.output_layer
        # bias L3
        grad_net.filter_bias2 = np.sum(grad_net.filtered_layer2)
        # bias L1
        grad_net.filter_bias1 = np.sum(grad_net.filtered_layer1)

        grad_net.scale_net(-1)
        return grad_net


    def train(self, batch_size=1, learn_rate=1, epochs=5, init=True):
        if init:
            self.random_init()
        for k in range(epochs):
            shuffle_together(train_imgs, train_labels)
            avg_cost = 0.0
            # divide the training set into (10000/batch_size) batches
            for i in range(10000 // batch_size):
                # classify a batch of images
                self.classify(train_imgs[i * batch_size:i * batch_size + batch_size], train_labels[i * batch_size:i * batch_size + batch_size])
                # calculate the average cost for the batch
                avg_cost += self.get_cost()
                # calculate the gradient vector via backprop(),
                # then scale the gradient by learn_rate and add to the net
                self.add_nets(self.backprop().scale_net(learn_rate))
            # average over the number of batches
            avg_cost /= (10000/batch_size)
            print('%.5f' % avg_cost, "\tepoch", k+1)


    def test(self):
        avg_cost = 0.0
        cm = np.zeros((10, 10), dtype=int)
        for i in range(10000):
            self.classify(test_imgs[i], test_labels[i])
            avg_cost += self.get_cost()
            cm[np.argmax(test_labels[i]), np.argmax(self.output_layer)] += 1
        avg_cost /= 10000
        accuracy = cm.trace() / 10000
        print("Confusion Matrix:")
        print(cm)
        print("average cost: ", avg_cost)
        print("accuracy: ", accuracy * 100, "%")


    def show_filters(self):
        for i in range(self.filter1.shape[0]):
            plt.figure()
            plt.imshow(self.filter1[i], cmap="Greys")
