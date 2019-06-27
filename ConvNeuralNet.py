import gzip
import numpy as np
import numpy.linalg as la
import scipy.signal as sg
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=300)
image_size = 28  # width and length
num_train_images = 60000
num_test_images = 10000
image_pixels = image_size * image_size
data_path = ""  # set path to image data if needed


def load_imgs(filename, num_imgs, image_size):
    f = gzip.open(filename, 'r')
    f.read(16)
    buf = f.read(image_size * image_size * num_imgs)
    imgs = (np.frombuffer(buf, dtype=np.uint8).astype(np.float32)).reshape(num_imgs, image_size**2)
    return imgs * (.99 / 255) + .01  # convert each pixel value into range [.01,1]

def load_labels(filename, num_labels):
    f = gzip.open(filename, 'r')
    f.read(8)
    buf = f.read(num_labels)
    labels = (np.frombuffer(buf, dtype=np.uint8).astype(np.int64)).reshape(num_labels, 1)
    return (np.arange(10) == labels).astype(np.float) # transform into one-hot representation


# loading the data
train_imgs = load_imgs(data_path + 'train-images-idx3-ubyte.gz', num_train_images, image_size)
test_imgs = load_imgs(data_path + 't10k-images-idx3-ubyte.gz', num_test_images, image_size)
train_labels = load_labels(data_path + 'train-labels-idx1-ubyte.gz', num_train_images)
test_labels = load_labels(data_path + 't10k-labels-idx1-ubyte.gz', num_test_images)


def sigmoid(input_vector):
    return np.arctan(input_vector) / np.pi + .5


def deriv_sigmoid(input_vector):
    return 1. / (np.pi * (1 + input_vector ** 2))


def soft_max(input_vector):
    return np.exp(input_vector) / np.sum(np.exp(input_vector))


def dropout(A, factor):
    Z = np.random.random_sample(A.shape)
    A[Z<factor] = 0.
    A *= 1/(1-factor)


def max_pool(A, nrow = 2, ncol = 2):
    return A.reshape(A.shape[0],A.shape[-2] // nrow, nrow, A.shape[-1] // ncol, ncol).max(axis=(-3,-1))
    # return A.reshape(A.shape[0] // nrow, nrow, A.shape[1] // ncol, ncol).max(axis=(1,3))


def inverse_pool(A, nrow = 2, ncol = 2):
    return np.kron(A, np.ones((nrow, ncol)) * 1/(nrow*ncol))


def pad(A, width):
    dim = list(A.shape)
    dim[-2], dim[-1] = dim[-2]+2*width, dim[-1]+2*width
    Z = np.zeros(dim)
    Z[...,width:A.shape[-2]+width,width:A.shape[-1]+width] = A
    return Z

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
        self.padded_input = np.zeros((32,32))
        self.filtered_layer1 = np.zeros((num_filters,28,28))
        self.pooled_layer1 = np.zeros((num_filters,14,14))
        self.padded_pool1 = np.zeros((num_filters,18,18))
        self.filtered_layer2 = np.zeros((num_filters,14,14))
        self.pooled_layer2 = np.zeros((num_filters,7, 7))
        self.output_layer = np.zeros((10,1))

        self.filter1 = np.zeros((num_filters,5,5))
        self.filter_bias1 = np.zeros((self.num_filters,1,1))
        self.filter2 = np.zeros((num_filters,num_filters,5,5))
        self.filter_bias2 = np.zeros((self.num_filters,1,1))
        self.weights = np.zeros((10, self.pooled_layer2.size))
        self.biases = np.zeros((10,1))
        self.label = np.zeros((10,1))


    # each weight is adjusted by 2 / sqrt(input size)
    def random_init(self):
        self.filter1 = np.random.standard_normal(self.filter1.shape)*np.sqrt(2/self.filter1[0].size)
        self.filter_bias1 = np.zeros(self.filter_bias1.shape)
        self.filter2 = np.random.standard_normal(self.filter2.shape)*np.sqrt(2/self.filter2[0].size)
        self.filter_bias2 = np.zeros(self.filter_bias2.shape)
        self.weights = np.random.standard_normal(self.weights.shape)*np.sqrt(2/self.weights.shape[-1])
        self.biases = np.zeros(self.biases.shape)

    def get_cost(self):
        return -np.log(self.output_layer[np.argmax(self.label)])


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
        self.filtered_layer1 = np.array([sg.correlate(self.padded_input,filt,mode="valid") for filt in self.filter1]) + self.filter_bias1
        self.filtered_layer1[self.filtered_layer1<0] = 0. # RELU
        # pooling 1
        self.pooled_layer1 = max_pool(self.filtered_layer1)
        self.padded_pool1 = pad(self.pooled_layer1, 2)
        # filtered layer 2
        self.filtered_layer2 = np.reshape(np.array([sg.correlate(self.padded_pool1, filt,mode="valid") for filt in self.filter2]),self.filtered_layer2.shape) + self.filter_bias2
        self.filtered_layer2[self.filtered_layer2<0] = 0. # RELU
        # pooling 2
        self.pooled_layer2 = max_pool(self.filtered_layer2)
        # output layer
        self.output_layer = soft_max((self.weights @ self.pooled_layer2.reshape(self.pooled_layer2.size,1)) + self.biases)
        self.label = input_label.reshape(input_label.size,1)



    def backprop(self):
        grad_net = CNN(self.num_filters)
        # error L5
        grad_net.output_layer = self.output_layer - self.label
        # errors L4
        grad_net.pooled_layer2 = np.reshape((self.weights.T @ grad_net.output_layer), self.pooled_layer2.shape)
        grad_net.pooled_layer2[self.pooled_layer2<=0] = 0.  # backprop through RELU
        # errors L3
        grad_net.filtered_layer2 = inverse_pool(grad_net.pooled_layer2)
        # errors L2
        grad_net.pooled_layer1 = np.reshape(np.array([sg.convolve(pad(filt, 11)[...,::-1,::-1], grad_net.filtered_layer2, mode='valid') for filt in self.filter2]), self.pooled_layer1.shape)
        grad_net.pooled_layer1[self.pooled_layer1<=0] = 0.  # backprop through RELU
        # errors L1
        grad_net.filtered_layer1 = inverse_pool(grad_net.pooled_layer1)

        # weights
        # weights L5
        grad_net.weights = np.outer(grad_net.output_layer, self.pooled_layer2.reshape(self.pooled_layer2.size))
        # weights L3
        for i in np.arange(grad_net.filtered_layer2.shape[0]):
            for j in np.arange(self.padded_pool1.shape[0]):
                grad_net.filter2[i][j] = sg.correlate(self.padded_pool1[j], grad_net.filtered_layer2[i],mode='valid')
        # weights L1
        for i in np.arange(grad_net.filtered_layer1.shape[0]):
            grad_net.filter1[i] = sg.correlate(self.padded_input, grad_net.filtered_layer1[i], mode='valid')
        # biases
        # biases L5
        grad_net.biases = grad_net.output_layer
        # bias L3
        grad_net.filter_bias2 = np.reshape(np.sum(grad_net.filtered_layer2,axis=(-1,-2)),grad_net.filter_bias2.shape)
        # bias L1
        grad_net.filter_bias1 = np.reshape(np.sum(grad_net.filtered_layer1, axis=(-1,-2)),grad_net.filter_bias1.shape)

        grad_net.scale_net(-1)
        return grad_net



    def train(self, epochs, learn_rate=.07, batch_size=10, init=True):
        if init:
            self.random_init()
        for k in range(epochs):
            shuffle_together(train_imgs, train_labels)
            avg_cost = 0.0
            # divide the training set into (10000/batch_size) batches
            for i in range(train_imgs.shape[0] // batch_size):
                grad = CNN(self.num_filters)
                for j in range(batch_size):
                    # classify a batch of images
                    self.classify(train_imgs[i*batch_size+j], train_labels[i*batch_size+j])
                    # calculate the average cost for the batch
                    avg_cost += self.get_cost()
                    # calculate the gradient vector via backprop()
                    grad.add_nets(self.backprop())
                # then scale the gradient by learn_rate and add to the net
                self.add_nets(grad.scale_net(learn_rate/batch_size))
            # average over the number of batches
            avg_cost /= (train_imgs.shape[0])
            print('%.5f' % avg_cost, "\tepoch", k+1)



    def test(self):
        avg_cost = 0.0
        cm = np.zeros((10, 10), dtype=int)
        for i in range(test_imgs.shape[0]):
            self.classify(test_imgs[i], test_labels[i])
            avg_cost += self.get_cost()
            cm[np.argmax(test_labels[i]), np.argmax(self.output_layer)] += 1
        avg_cost /= test_imgs.shape[0]
        accuracy = cm.trace() / test_imgs.shape[0]
        print("Confusion Matrix:")
        print(cm)
        print("average cost: ", avg_cost)
        print("accuracy: ", accuracy * 100, "%")



    def show_filters(self):
        for i in range(self.filter1.shape[0]):
            plt.figure()
            plt.imshow(self.filter1[i], cmap="Greys")

