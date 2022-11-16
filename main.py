import copy
import gzip
import math
import os
import pickle
import sys
import time

import numba
from dataclasses import dataclass

import numpy as np
from numba import njit
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

        neurons_full_shape = (*neuron_layers_shape, nodes)
        output_full_shape = (outputs, neuron_layers_shape[1])
        # print(neurons_full_shape, output_full_shape)
        self.init_values = (neuron_layers_shape, neuron_bias, outputs, output_bias, nodes,
                            neurons_full_shape, output_full_shape)

        self.neurons_weights = numpy.random.rand(*neurons_full_shape)
        self.neurons_biases = np.full(neurons_full_shape, neuron_bias)
        self.output_weights = numpy.random.rand(*output_full_shape)
        self.output_biases = np.full(output_full_shape, output_bias)

        self.avg_error = 0

    def get_weights(self):
        return self.neurons_weights, self.output_weights
    def set_weights(self, neuron_weights, output_weights):
        self.neurons_weights = neuron_weights
        self.output_weights = output_weights

    def regenerate_weights(self):
        self.neurons_weights = numpy.random.rand(*self.init_values[-2])
        self.output_weights = numpy.random.rand(*self.init_values[-1])

    def get_first_layer_weights(self, a):
        r = numpy.zeros(self.neurons_weights.shape[1])
        # print(a, self.neurons_weights[0], self.neurons_biases)
        for i, (b, w) in enumerate(zip(self.neurons_biases[0], self.neurons_weights[0])):
            # print(a, w)
            # print(b, w, np.dot(w, a))
            # print(np.dot(w, a))
            # print(self.norm_func(np.dot(w, a) + b))
            r[i] = self.norm_func((np.dot(w, a) + b)[0])
        return r

    # TODO: Norm the output
    def get_output_layer_weights(self, a):
        r = numpy.zeros(self.output_weights.shape[0])
        for i, (b, w) in enumerate(zip(self.output_biases, self.output_weights)):
            # print(a, w)
            # print(b, w, np.dot(w, a))
            # print(np.dot(w, a))
            # print(self.norm_func(np.dot(w, a) + b))
            r[i] = self.norm_func((np.dot(w, a) + b)[0])
        s = r.sum()
        r /= s
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
        for i, image in enumerate(data_points):  # self.train_images[:data_points]
            r = self.get_first_layer_weights(image.flat[:])
            o = self.get_output_layer_weights(r)
            # print(o)
            error_vec[i] = self.get_error(o, numbers[i])  # self.train_numbers

        return np.average(error_vec)

    # TODO: NEEDS FIX
    def mutate(self, learning_rate):
        dif_arr_n = numpy.random.uniform(-learning_rate, learning_rate, self.neurons_weights.size) \
            .reshape(self.neurons_weights.shape)
        # dif_arr_n.reshape(self.neurons_weights.shape)

        self.neurons_weights += dif_arr_n

        dif_arr_o = numpy.random.uniform(-learning_rate, learning_rate, self.output_weights.size) \
            .reshape(self.output_weights.shape)
        # dif_arr_o.reshape(self.output_weights.shape)

        self.output_weights += dif_arr_o

    def get_mutated_copy(self, learning_rate):
        ncopy = copy.deepcopy(self)
        ncopy.mutate(learning_rate)
        return ncopy

    def compute_number(self, image):
        r = self.get_first_layer_weights(image.flat[:])
        o = self.get_output_layer_weights(r)

        # print(o)
        return np.where(o == o.max())


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
        (train_images, train_numbers), (test_images, test_numbers) = self.get_data_from_file()

        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)

        return (train_images, train_numbers), (test_images, test_numbers)

    def evolve(self, networks, population, learning_rate):
        best_multiple = population // 20
        best = networks[:best_multiple]
        new = copy.deepcopy(best)
        [print({net.avg_error}, end=" ") for net in best]
        # print()
        for i, net in enumerate(best):
            # 5 - 30 25 20 15 10
            for _ in range(best_multiple * (6 - i) - 1):
                new.append(net.get_mutated_copy(learning_rate))

        # [print({net.avg_error}, end=" ") for net in best]
        # print()

        return new

    def step_decay(self, gen, init_learning_rate=0.2, drop=0.8, gen_drop=10):
        learning_rate = init_learning_rate * math.pow(drop, math.floor((1 + gen) / gen_drop))
        return learning_rate

    # @njit(parallel=True)
    def train(self, population, gens, data_points, learning_rate, init_tests):
        self.generate_better_random_net(init_tests)

        print("Generated better random network")

        r, w = self.compare_to_test((0, 10000), False)
        print(self.data.avg_error)
        print(f"Right: {r}  -   Wrong: {w}      (Untrained - compared to TRAIN data)")

        r, w = self.compare_to_test((0, 10000))
        print(self.data.avg_error)
        print(f"Right: {r}  -   Wrong: {w}      (Untrained - compared to TEST data)")

        networks = [copy.deepcopy(self.data) for _ in range(population)]
        # networks[0].set_weights(self.data.get_weights())
        # print(networks)
        # init_learning_rate = learning_rate
        left, right = data_points
        for gen in range(gens):
            # learning_rate = self.step_decay(gen, init_learning_rate)
            start = time.time()
            print(f"{gen}", end=" ")
            for i, network in enumerate(networks):
                networks[i].set_avg_error(self.train_images[left:right], self.train_numbers)
                # print(networks[i].avg_error)

            # TODO: Decide where this should be
            networks.sort(key=lambda x: x.avg_error)
            networks = self.evolve(networks, population, learning_rate)

            print(f"({time.time() - start:.2}s)")

            # [print(net.avg_error, end=" ") for net in networks]
            # print()
        self.data = networks[0]

    def compare_to_test(self, in_range, test=True):
        right = 0
        wrong = 0

        f, t = in_range

        images, numbers = self.test_images[f:t], self.test_numbers[f:t]
        if not test:
            images, numbers = self.train_images[f:t], self.train_numbers[f:t]

        for image, num in zip(images, numbers):
            comp_num = self.data.compute_number(image)
            if comp_num == num:
                right += 1
            else:
                wrong += 1

        return right, wrong

    def generate_better_random_net(self, tests, comparisons=100):
        arr = []
        for i in range(tests):
            # print(i)
            arr.append([self.compare_to_test((0, comparisons), False)[0], copy.deepcopy(self.data)])
            self.data.regenerate_weights()
        self.data = max(arr, key=lambda x: x[0])[1]

    def save_weights(self, map_name="net"):
        cur_dir = os.path.dirname(__file__)
        # print(cur_dir)
        full_path = f"{cur_dir}/data/saved_weights/{map_name}"
        print(full_path)
        try:
            os.mkdir(full_path)
            with open(f"{full_path}/neuron_weights.npy", "x"):
                pass
            with open(f"{full_path}/output_weights.npy", "x"):
                pass
        except OSError as e:
            pass

        numpy.save(f"{full_path}/neuron_weights.npy", self.data.neurons_weights)
        numpy.save(f"{full_path}/output_weights.npy", self.data.output_weights)

    def load_weights(self, map_name="net"):
        cur_dir = os.path.dirname(__file__)
        full_path = f"{cur_dir}/data/saved_weights/{map_name}"

        neuron_weights = numpy.load(f"{full_path}/neuron_weights.npy")
        output_weights = numpy.load(f"{full_path}/output_weights.npy")
        self.data.set_weights(neuron_weights, output_weights)

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

time_before = time.time()

n = Network(neuron_layers_shape=(1, 20), neuron_bias=-35, outputs=10, output_bias=-9, norm_func_name="expit")
n.load_weights("net10")
# data_start = copy.deepcopy(n.data)

# for _ in range(100):
# r, w = n.compare_to_test((0, 100))
# print(n.data.avg_error)
# print(f"Right: {r}  -   Wrong: {w}")

# n.data.regenerate_weights()

print("Network init")

# Generating better network doesn't matter, but might as well do it quickly
n.train(100, 200, (1000, 1200), 0.2, 100)
# data_end = copy.deepcopy(n.data)

r, w = n.compare_to_test((1000, 1200), False)
print(n.data.avg_error)
print(f"Right: {r}  -   Wrong: {w}      (Trained - compared to TRAIN data)")

r, w = n.compare_to_test((0, 10000))
print(n.data.avg_error)
print(f"Right: {r}  -   Wrong: {w}      (Trained - compared to TEST data)")

print(f"Total time: {time.time() - time_before:.2}s")

# net1 - 27%
# net2 - ~20?
# net3 -
n.save_weights("net11")

# n.data = data_start
# r, w = n.compare_to_test((0, 10000))
# print(n.data.avg_error)
# print(f"Right: {r}  -   Wrong: {w}")

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
