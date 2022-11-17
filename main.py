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
from line_profiler_pycharm import profile
from numba import njit
from scipy.special import expit
import logging, os
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from keras.datasets import mnist

import numpy
from matplotlib import pyplot

# @njit
def normalize(data, divider=255):
    return data * (1 / divider)

@njit
def get_error(output, number):
    num_vec = numpy.zeros(output.size)
    num_vec[number] = 1.
    result = (output - num_vec)
    # print(result)
    result *= result
    # print(result)
    return np.sum(result)


@njit("float64[:](float64[::1], float64[:,::1], int64[::1])")
def get_output_layer_weights(a, output_weights, output_biases):
    r = numpy.zeros(output_weights.shape[0])
    for i, (b, w) in enumerate(zip(output_biases, output_weights)):
        # print(a, w)
        # print(b, w, np.dot(w, a))
        # print(np.dot(w, a))
        # print(self.norm_func(np.dot(w, a) + b))
        r[i] = 1 / (1 + np.exp(-(np.dot(np.ascontiguousarray(w), a) + b)))

    # r = np.dot(output_weights, a) + output_biases
    # r = get_layer_weights(a, output_weights, output_biases)

    s = r.sum()
    r /= s
    return r


@njit("float64[:](float64[::1], float64[:,::1], int64[::1])")
# # @profile
def get_layer_weights(a, neuron_weights, neuron_biases):
    r = numpy.zeros(neuron_weights.shape[0])
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))
    # print(a, self.neurons_weights[0], self.neurons_biases)

    for i, (b, w) in enumerate(zip(neuron_biases, neuron_weights)):
        # print(a, w)
        # print(b, w, np.dot(w, a))
        # print(np.dot(w, a))
        # print(self.norm_func(np.dot(w, a) + b))
        # sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # 1 / (1 + np.exp(-(np.dot(w, a) + b)[0]))
        # d = np.dot(w, a)
        r[i] = 1 / (1 + np.exp(-(np.dot(np.ascontiguousarray(w), a) + b)))

    # dot = np.dot(neuron_weights, a)
    # r = 1 / (1 + np.exp(-(np.dot(neuron_weights, a) + neuron_biases)))
    return r


@njit
def step_decay(gen, init_learning_rate=0.2, drop=0.99, gen_drop=10):
        learning_rate = init_learning_rate * math.pow(drop, math.floor((1 + gen) / gen_drop))
        return learning_rate


# @njit
def evolve_best(networks, population, learning_rate):
    best = networks[0]
    new = [best]
    print(f"[{best.avg_error}]", end=" ")

    for _ in range(population - 1):
        new.append(best.get_mutated_copy(learning_rate))

    # [print({net.avg_error}, end=" ") for net in best]
    # print()

    return new


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
        self.neurons_biases = np.full(neuron_layers_shape, neuron_bias, dtype=np.int64)
        self.output_weights = numpy.random.rand(*output_full_shape)
        self.output_biases = np.full(outputs, output_bias, dtype=np.int64)

        self.avg_error = 0

    def get_weights(self):
        return self.neurons_weights, self.output_weights

    def set_weights(self, neuron_weights, output_weights):
        self.neurons_weights = neuron_weights
        self.output_weights = output_weights

    def regenerate_weights(self):
        self.neurons_weights = numpy.random.rand(*self.init_values[-2])
        self.output_weights = numpy.random.rand(*self.init_values[-1])

    # # @profile
    # @njit(parallel=True)
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

    # # @profile
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

    # @profile
    def get_avg_error(self, data_points, numbers):
        # print(data_points.)
        error_vec = numpy.zeros(data_points.shape[0])
        for i, image in enumerate(data_points):  # self.train_images[:data_points]
            # r = self.get_first_layer_weights(image.flat[:])
            r = get_layer_weights(image.flat[:], self.neurons_weights[0],
                                  self.neurons_biases[0])
            o = get_output_layer_weights(r, self.output_weights,
                                         self.output_biases)
            # print(o)
            error_vec[i] = get_error(o, numbers[i])  # self.train_numbers

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

    # # @profile
    def compute_number(self, image):
        r = get_layer_weights(image.flat[:], self.neurons_weights[0],
                              self.neurons_biases[0])
        o = get_output_layer_weights(r, self.output_weights,
                                     self.output_biases)

        # print(o)
        return np.argmax(o)


class Network:
    # # @profile
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

    # @profile
    def get_data_from_file(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')
        f.close()

        # loading the dataset
        return data

    # # @profile
    def get_data_normalized(self):
        # (train_images, train_numbers), (test_images, test_numbers) = mnist.load_data()
        (train_images, train_numbers), (test_images, test_numbers) = self.get_data_from_file()

        trn_i = train_images.astype(numpy.double)
        tst_i = test_images.astype(numpy.double)

        # train_numbers = train_numbers.astype(numpy.int64)
        # test_numbers = test_numbers.astype(numpy.int64)

        train_images = normalize(trn_i)
        test_images = normalize(tst_i)

        return (train_images, train_numbers), (test_images, test_numbers)

    def evolve(self, networks, population, learning_rate):
        best_multiple = population // 20
        best = networks[:best_multiple]
        new = copy.deepcopy(best)
        [print(f"[{net.avg_error}] ", end=" ") for net in best]
        # print()
        for i, net in enumerate(best):
            # 5 - 30 25 20 15 10
            for _ in range(best_multiple * (6 - i) - 1):
                new.append(net.get_mutated_copy(learning_rate))

        # [print({net.avg_error}, end=" ") for net in best]
        # print()

        return new

    # @njit(parallel=True)
    # @profile
    def train(self, population, gens, data_points, learning_rate, init_tests):
        # self.generate_better_random_net(init_tests)

        # print("Generated better random network")

        r, w = self.compare_to_test(data_points, False)
        print(self.data.avg_error)
        print(f"Right: {r}  -   Wrong: {w}      (Untrained - compared to TRAIN data)")

        r, w = self.compare_to_test((0, 10000))
        print(self.data.avg_error)
        print(f"Right: {r}  -   Wrong: {w}      (Untrained - compared to TEST data)")

        networks = [copy.deepcopy(self.data) for _ in range(population)]
        # networks[0].set_weights(self.data.get_weights())
        # print(networks)
        init_learning_rate = learning_rate
        left, right = data_points
        for gen in range(gens):
            # Step decay:
            #   - Low gens: drop = 0.95-0.99, drop_gens = 10
            #   - High gens: drop = 0.99-0.995, drop_gens = 100
            learning_rate = step_decay(gen, init_learning_rate, 0.99, 100)

            start = time.time()
            print(f"{gen + 1}/{gens}:", end=" ")
            for i, network in enumerate(networks):
                networks[i].set_avg_error(self.train_images[left:right], self.train_numbers)
                # print(networks[i].avg_error)

            # TODO: Decide where this should be
            networks.sort(key=lambda x: x.avg_error)
            networks = evolve_best(networks, population, learning_rate)
            # print(f"[{networks[0].avg_error}]", end=" ")

            print(f"({time.time() - start:.2f}s)")

            # [print(net.avg_error, end=" ") for net in networks]
            # print()
            self.data = networks[0]

    def compare_to_test_data(self, in_range=(0, 60000), test=True):
        size = in_range[1] - in_range[0]
        right_wgt = np.zeros((size, *self.train_images.shape[1:]), dtype=np.double)
        right_num = np.full(size, -1, dtype=np.int64)
        wrong_wgt = np.zeros((size, *self.train_images.shape[1:]), dtype=np.double)
        wrong_num = np.full(size, -1, dtype=np.int64)

        f, t = in_range

        images, numbers = self.test_images[f:t], self.test_numbers[f:t]
        if not test:
            images, numbers = self.train_images[f:t], self.train_numbers[f:t]

        for i, (image, num) in enumerate(zip(images, numbers)):
            comp_num = self.data.compute_number(image)
            if comp_num == num:
                right_wgt[i] = image
                right_num[i] = num
            else:
                wrong_wgt[i] = image
                wrong_num[i] = num

        rw = right_wgt[~np.all(right_wgt == 0, axis=(1, 2))]
        rn = right_num[right_num != -1]
        ww = wrong_wgt[~np.all(wrong_wgt == 0, axis=(1, 2))]
        wn = wrong_num[wrong_num != -1]
        return rw, rn, ww, wn

    # # @profile
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

        # print("Generated better random network")

    def save_weights(self, map_name="net"):
        cur_dir = os.path.dirname(__file__)
        # print(cur_dir)
        full_path_dir = f"{cur_dir}/data/saved_nets/{map_name}"
        full_path_net = f"{full_path_dir}/net"
        print(full_path_net)
        try:
            try:
                os.mkdir(full_path_dir)
            except OSError as e:
                pass
            os.mkdir(full_path_net)

            with open(f"{full_path_net}/neuron_weights.npy", "x"):
                pass
            with open(f"{full_path_net}/output_weights.npy", "x"):
                pass
        except OSError as e:
            print(e)

        numpy.save(f"{full_path_net}/neuron_weights.npy", self.data.neurons_weights)
        numpy.save(f"{full_path_net}/output_weights.npy", self.data.output_weights)

    def save_right_wrong_data(self, map_name="net"):
        cur_dir = os.path.dirname(__file__)
        # print(cur_dir)
        full_path_dir = f"{cur_dir}/data/saved_nets/{map_name}"
        full_path_rwd = f"{full_path_dir}/right_wrong_data"
        print(full_path_rwd)
        try:
            try:
                os.mkdir(full_path_dir)
            except OSError as e:
                pass
            os.mkdir(full_path_rwd)

            with open(f"{full_path_rwd}/right_weight.npy", "x"):
                pass
            with open(f"{full_path_rwd}/right_num.npy", "x"):
                pass
            with open(f"{full_path_rwd}/wrong_weight.npy", "x"):
                pass
            with open(f"{full_path_rwd}/wrong_num.npy", "x"):
                pass
        except OSError as e:
            print(e)

        rw, rn, ww, wn = self.compare_to_test_data()

        numpy.save(f"{full_path_rwd}/right_weight.npy", rw)
        numpy.save(f"{full_path_rwd}/right_num.npy", rn)
        numpy.save(f"{full_path_rwd}/wrong_weight.npy", ww)
        numpy.save(f"{full_path_rwd}/wrong_num.npy", wn)

    def load_right_wrong_data(self, map_name="net"):
        cur_dir = os.path.dirname(__file__)
        full_path = f"{cur_dir}/data/saved_nets/{map_name}/right_wrong_data"

        rw = numpy.load(f"{full_path}/right_weight.npy")
        rn = numpy.load(f"{full_path}/right_num.npy")
        ww = numpy.load(f"{full_path}/wrong_weight.npy")
        wn = numpy.load(f"{full_path}/wrong_num.npy")

        return rw, rn, ww, wn

    def load_weights(self, map_name="net"):
        print(f"Dir: {map_name}")

        cur_dir = os.path.dirname(__file__)
        full_path = f"{cur_dir}/data/saved_nets/{map_name}/net"

        neuron_weights = numpy.load(f"{full_path}/neuron_weights.npy")
        output_weights = numpy.load(f"{full_path}/output_weights.npy")
        self.data.set_weights(neuron_weights, output_weights)


# @profile
def main():
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

    # load_dir = "net_20w_100_10k"
    load_dir = "net_20w_100_20k"              # 50627 - 8384
    # load_dir = "net_20w_100_20k_20k_0-005lr"
    # load_dir = "net_20w_100_20k_20k_0-01lr"
    # load_dir = "net_20w_100_20k_60k_0-05lr"
    # load_dir = "net_20w_100_60k"
    # load_dir = "net_20w_100_20k_test"           # 50620 - 8388

    # load_dir = "net_50w_1"
    # load_dir = "net_75w_1"
    # load_dir = "net_100w_1"

    # save_dir = "net_20w_100_20k"
    # save_dir = "net_20w_100_20k_20k_0-005lr"
    # save_dir = "net_20w_100_20k_20k_0-01lr"
    # save_dir = "net_20w_100_20k_60k_0-02lr"
    # save_dir = "net_20w_100_20k_test"

    n.load_weights(load_dir)

    # r_w = n.load_right_wrong_data(load_dir)

    # for _ in range(100):
    # r, w = n.compare_to_test((0, 100))
    # print(n.data.avg_error)
    # print(f"Right: {r}  -   Wrong: {w}")

    print("Network init")
    print(f"Shape: {n.data.neurons_weights.shape}")

    # Generating better network doesn't matter, but might as well do it quickly
    data_points = (0, 60000)

    # n.compare_to_test_data(data_points)

    try:
        # n.train(10, 3500, data_points, 0.02, 100)
        pass
    except:
        pass
    finally:
        r, w = n.compare_to_test(data_points, False)
        print(n.data.avg_error)
        print(f"Right: {r}  -   Wrong: {w}      (Trained - compared to TRAIN data)")

        r, w = n.compare_to_test((0, 10000))
        print(n.data.avg_error)
        print(f"Right: {r}  -   Wrong: {w}      (Trained - compared to TEST data)")

        print(f"Total time: {time.time() - time_before:.2f}s")

        # n.save_weights(save_dir)
        # n.save_right_wrong_data(save_dir)

    # net1 - 27%
    # net2 - ~20?
    # net7 - 60% test, 80% training (91% on 100 first)
    # net12 - 53% test, 97% training (100), error - 0.045672393817552004

    # n.save_weights(save_dir)
    # n.save_right_wrong_data(save_dir)

    # for i in range(9):
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    # pyplot.show()


if __name__ == "__main__":
    main()

