import gzip
import pickle
import sys

import numpy as np
from scipy.special import expit

import numpy
from matplotlib import pyplot


f = gzip.open('data/mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()

# loading the dataset
(train_images, train_numbers), (test_images, test_numbers) = data

train_images_float = train_images.astype(numpy.float32)
train_images_float *= 1/255
# print(train_images_float[0])
# (tx, ty) = train_images_float.shape


t = (1, 20, train_images_float[0].size)
weights = numpy.random.rand(*t)
# print(weights.shape)


class Network:
    def __init__(self, neuron_layers_shape: tuple, neuron_layers_biases: numpy.array,
                 output_layers_shape: tuple, output_layer_biases: numpy.array,
                 norm_func_name: str = "expit"):
        self.neurons_weights = numpy.random.rand(*neuron_layers_shape)
        self.neurons_biases = neuron_layers_biases
        self.output_weights = numpy.random.rand(*output_layers_shape)
        self.output_biases = output_layer_biases
        self.norm_func = {
            "expit": expit,
        }.get(norm_func_name.lower().strip(), "expit")

    def get_first_layer_weights(self, a):
        r = numpy.zeros(20)
        print(a, self.neurons_weights[0], self.neurons_biases)
        for i, (b, w) in enumerate(zip(self.neurons_biases[0], self.neurons_weights[0])):
            # print(a, w)
            # print(b, w, np.dot(w, a))
            print(np.dot(w, a))
            # print(self.norm_func(np.dot(w, a) + b))
            r[i] = self.norm_func(np.dot(w, a) + b)[0]
        return r

    def get_output_layer_weights(self, a):
        r = numpy.zeros(10)
        for i, (b, w) in enumerate(zip(self.output_biases, self.output_weights)):
            # print(a, w)
            # print(b, w, np.dot(w, a))
            print(np.dot(w, a))
            # print(self.norm_func(np.dot(w, a) + b))
            r[i] = self.norm_func((np.dot(w, a) + b)[0])
        return r


neurons = 20
layers = 1
neurons_shape = (layers, neurons, train_images_float[0].size)
neurons_biases = numpy.zeros(neurons_shape)
neurons_biases.fill(-52)

outputs = 10
output_shape = (outputs, neurons)
output_biases = numpy.zeros(output_shape)
output_biases.fill(-8)

n = Network(neurons_shape, neurons_biases, output_shape, output_biases, "expit")

# print(train_images_float[0].flat[:].shape)

r = n.get_first_layer_weights(train_images_float[0].flat[:])
o = n.get_output_layer_weights(r)

print(o)

# for v in r:
#     print(v)

# print(output)

# gens = 100
# # for gen in range(gens):
# for image in test_images[:100]:
#
#     pass

# print(train_images.)

# printing the shapes of the vectors
# print('X_train: ' + str(train_x.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  ' + str(test_x.shape))
# print('Y_test:  '  + str(test_y.shape))

# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()
