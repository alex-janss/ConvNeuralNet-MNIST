import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


np.set_printoptions(linewidth=300)
image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "C:/Users/Alex/PycharmProjects/ExperimentBox/"
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

# shuffles a pair of vectors together,
# i.e. both vectors are shuffled exactly the same way
def shuffle_together(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

class CNN:

    def __init__(self, num_filters=3):
        self.num_filters = num_filters
        self.input_layer = np.zeros((28,28))
        self.filtered_layer = np.zeros((self.num_filters,24,24))
        self.filtered_z = np.zeros((self.num_filters,24,24))
        self.pooled_layer = np.zeros((self.num_filters,12,12))
        self.pooled_z = np.zeros((self.num_filters,12,12))
        self.output_layer = np.zeros((10,1))
        self.output_z = np.zeros((10,1))

        self.filter = np.zeros((self.num_filters,5,5))
        self.filter_bias = np.zeros(self.num_filters)
        self.weights = np.zeros((10,12*12*self.num_filters))
        self.biases = np.zeros((10,1))
        self.errors = np.zeros((10,1))


    def random_init(self):
        self.filter = np.random.randn(self.num_filters,5,5)
        self.filter_bias = np.random.randn(self.num_filters)
        self.weights = np.random.randn(10,12*12*self.num_filters)
        self.biases = np.random.randn(10,1)

    def get_cost(self):
        return la.norm(self.errors)

    def add_nets(self, input_net):
        self.filter += input_net.filter
        self.filter_bias += input_net.filter_bias
        self.weights += input_net.weights
        self.biases += input_net.biases

    def scale_net(self, factor):
        self.filter *= factor
        self.filter_bias *= factor
        self.weights  *= factor
        self.biases *= factor
        return self


    def classify(self, input_img, input_label):
        self.input_layer = input_img.reshape(28,28)
        for i in range(self.num_filters):
            self.filtered_z[i] = correlate2d(self.input_layer,self.filter[i],mode="valid") + self.filter_bias[i]
        self.filtered_layer = sigmoid(self.filtered_z)
        for i in range(self.num_filters):
            self.pooled_z[i] = max_pool(self.filtered_z[i])
        self.pooled_layer = sigmoid(self.pooled_z)
        self.output_z = (self.weights @ self.pooled_layer.reshape(self.pooled_layer.size,1)) + self.biases
        self.output_layer = sigmoid(self.output_z)
        self.errors = self.output_layer - input_label.reshape(input_label.size,1)


    def backprop(self):
        grad_net = CNN(self.num_filters)
        # errors L3
        grad_net.output_layer = 2 * self.errors * deriv_sigmoid(self.output_z)
        # biases L3
        grad_net.biases = grad_net.output_layer
        # weights L3
        grad_net.weights = np.outer(grad_net.output_layer, self.pooled_layer.reshape(self.pooled_layer.size))
        # errors L2
        grad_net.pooled_layer = (self.weights.T @ grad_net.output_layer).reshape(self.num_filters,12,12) * deriv_sigmoid(self.pooled_z)
        # errors L1
        for i in range(self.num_filters):
            grad_net.filtered_layer[i] = inverse_pool(grad_net.pooled_layer[i])
        # bias L1
        grad_net.filter_bias = np.sum(grad_net.filtered_layer, axis=(1,2))
        # filter weights L1
        for i in range(self.num_filters):
            grad_net.filter[i] = correlate2d(self.input_layer, grad_net.filtered_layer[i], mode="valid")
        grad_net.scale_net(-1)
        return grad_net


    def train(self, batch_size, learn_rate, epochs=5, init=True):
        if init:
            self.random_init()
        for k in range(epochs):
            shuffle_together(train_imgs, train_labels)
            avg_cost = 0.0
            # divide the training set into (10000/batch_size) batches
            for i in range(int(10000 / batch_size)):
                # classify a batch of images
                self.classify(train_imgs[i * batch_size:i * batch_size + batch_size], train_labels[i * batch_size:i * batch_size + batch_size])
                # calculate the average cost for the batch
                avg_cost += self.get_cost()
                # calculate the gradient vector via backprop(),
                # then scale the gradient by learn_rate and add to the net
                self.add_nets(self.backprop().scale_net(learn_rate))
            # average over the number of batches
            avg_cost /= (10000/batch_size)
            print(avg_cost, "\tepoch", k+1)


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
        for i in range(self.filter.shape[0]):
            plt.figure()
            plt.imshow(self.filter[i], cmap="Greys")
