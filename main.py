import gzip
import pickle
import sys
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

import numpy
from matplotlib import pyplot

# train_images_float = train_images.astype(numpy.float32)
# train_images_float *= 1/255
# print(train_images_float[0])
# (tx, ty) = train_images_float.shape


# t = (1, 20, train_images_float[0].size)
# weights = numpy.random.rand(*t)
# print(weights.shape)

class NetworkData:
    def __init__(self, neuron_layers_shape: tuple, neuron_bias: int,
                 outputs: int, output_bias: int, nodes: int, norm_func):
        self.norm_func = norm_func

        # self.init_values = (neuron_layers_shape, neuron_bias, outputs, output_bias, nodes)

        neurons_full_shape = (*neuron_layers_shape, nodes)
        output_full_shape = (outputs, neuron_layers_shape[1])
        # print(neurons_full_shape, output_full_shape)

        self.neurons_weights = numpy.random.rand(*neurons_full_shape)
        self.neurons_biases = np.full(neurons_full_shape, neuron_bias)
        self.output_weights = numpy.random.rand(*output_full_shape)
        self.output_biases = np.full(output_full_shape, output_bias)

        self.avg_error = 0

    def get_first_layer_weights(self, a):
        r = numpy.zeros(20)
        # print(a, self.neurons_weights[0], self.neurons_biases)
        for i, (b, w) in enumerate(zip(self.neurons_biases[0], self.neurons_weights[0])):
            # print(a, w)
            # print(b, w, np.dot(w, a))
            # print(np.dot(w, a))
            # print(self.norm_func(np.dot(w, a) + b))
            r[i] = self.norm_func((np.dot(w, a) + b)[0])
        return r

    def get_output_layer_weights(self, a):
        r = numpy.zeros(10)
        for i, (b, w) in enumerate(zip(self.output_biases, self.output_weights)):
            # print(a, w)
            # print(b, w, np.dot(w, a))
            # print(np.dot(w, a))
            # print(self.norm_func(np.dot(w, a) + b))
            r[i] = self.norm_func((np.dot(w, a) + b)[0])
        return r

    def set_avg_error(self, data_points, numbers):
        self.avg_error = self.get_avg_error(data_points, numbers)

    def get_error(self, output, number):
        num_vec = numpy.zeros(output.size)
        num_vec[number] = 1.
        result = (output - num_vec)
        # print(result)
        result *= result
        # print(result)
        return np.sum(result)

    def get_avg_error(self, data_points, numbers):
        # print(data_points.)
        error_vec = numpy.zeros(data_points.shape[0])
        for i, image in enumerate(data_points):     # self.train_images[:data_points]
            r = self.get_first_layer_weights(image.flat[:])
            o = self.get_output_layer_weights(r)
            error_vec[i] = self.get_error(o, numbers[i])     # self.train_numbers

        return np.average(error_vec)

    def mutate(self, mutation_range):
        pass


class Network:
    def __init__(self, neuron_layers_shape: tuple, neuron_bias: int,
                 outputs: int, output_bias: int,
                 norm_func_name: str = "expit"):
        # self.init_values = (neuron_layers_shape, neuron_bias, outputs, output_bias, norm_func_name)

        self.image_data = (self.train_images, self.train_numbers), (self.test_images, self.test_numbers) = \
            self.get_data_normalized()

        self.norm_func = {
            "expit": expit,
        }.get(norm_func_name.lower().strip(), "expit")

        self.data_params = (neuron_layers_shape, neuron_bias, outputs, output_bias,
                            self.test_images[0].size, self.norm_func)

        self.data = NetworkData(*self.data_params)

    def get_data_from_file(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')
        f.close()

        # loading the dataset
        return data

    def normalize(self, data, divider=255):
        return data.astype(numpy.float32) * (1 / divider)

    def get_data_normalized(self):
        (train_images, train_numbers),  (test_images, test_numbers) = self.get_data_from_file()

        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)

        return (train_images, train_numbers) , (test_images, test_numbers)

    def evolve(self, networks, population, mutation_range):
        best = networks[:population // 20]
        new = []
        [print({net.avg_error}, end=" ") for net in best]
        print()
        # for pair in best:
        #     for i, weight in enumerate(pair[0].neurons_weights):
        #         new.append()


        return networks

    def train(self, population, gens, data_points, mutation_range):
        networks = [NetworkData(*self.data_params) for _ in range(population)]
        # print(networks)
        for gen in range(gens):
            for i, network in enumerate(networks):
                networks[i].set_avg_error(self.test_images[:data_points], self.train_numbers)
                # print(networks[i].avg_error)
            networks.sort(key=lambda x: x.avg_error)
            networks = self.evolve(networks, population, mutation_range)

            # print(networks)
            # print()



# neurons = 20
# layers = 1
# neurons_shape = (layers, neurons, train_images_float[0].size)
# neurons_biases = numpy.zeros(neurons_shape)
# neurons_biases.fill(-52)
#
# outputs = 10
# output_shape = (outputs, neurons)
# output_biases = numpy.zeros(output_shape)
# output_biases.fill(-8)

n = Network(neuron_layers_shape=(1, 20), neuron_bias=-52, outputs=10, output_bias=-8, norm_func_name="expit")
# print(n.get_average_error(100))
n.train(100, 100, 100, 0.2)

# print(train_images_float[0].flat[:].shape)

# r = n.get_first_layer_weights(train_images_float[0].flat[:])
# o = n.get_output_layer_weights(r)
# e = n.get_error(o, train_numbers[0])
#
# print()
# print(e)

# print(o)

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
