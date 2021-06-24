from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from time import time
from util.plots import show_plot

feature = pd.read_csv('../data/feature_nirspec100.csv').values
label = pd.read_csv('../data/label_nirspec100.csv').values

train_feature = feature[:5000]
train_label = label[:5000]
test_feature = feature[50000:]
test_label = label[50000:]


loss = keras.losses.MeanSquaredError()
layers_format = [(keras.layers.InputLayer, [198]),
                 (keras.layers.Dense, 150, 'relu'),
                 (keras.layers.Dropout, 0.1),
                 (keras.layers.Dense, 44, 'relu'),
                 (keras.layers.Dense, 12, 'relu')]
layers_available_for_evolution = (keras.layers.Dense,)


class Individual:
    def __init__(self, layers, loss, optimizer=keras.optimizers.Adam(clipvalue=2), weights=None, biases=None):
        self.model = keras.Sequential()

        layer_instance = layers[0][0](input_shape=layers[0][1])
        self.model.add(layer_instance)

        i = 0
        for layer_data in layers[1:]:
            if layer_data[0] == keras.layers.Dense:
                layer_instance = layer_data[0](units=layer_data[1], activation=layer_data[2],
                                               kernel_initializer=tf.initializers.constant(weights[i].numpy())
                                               if weights else 'glorot_uniform',
                                               bias_initializer=tf.initializers.constant(biases[i].numpy())
                                               if biases else 'zeros')
                i += 1
            elif layer_data[0] == keras.layers.Dropout:
                layer_instance = layer_data[0](rate=layer_data[1])

            self.model.add(layer_instance)

        self.model.compile(loss=loss, optimizer=optimizer)


def mutate(values, prob, interval=None, uniform=True):
    p = np.random.random(size=values.shape) < prob
    indices = np.argwhere(p)

    if uniform is not None:
        f = np.random.uniform if uniform else np.random.normal
        values_to_change = values.numpy()[p]
        random_values = f(*interval, size=indices.shape[0])
        values.scatter_nd_update(indices, values_to_change + random_values)
        return

    float64_p_values = values.numpy()[p].astype(np.float64)
    bit_mutate_values = np.array([[1 << i if random.random() < prob else 0 for i in range(64)]
                                 for _ in range(indices.shape[0])])

    if len(bit_mutate_values.shape) == 2:
        m = np.bitwise_xor.reduce(bit_mutate_values, axis=1, dtype=np.int64)
        new_values = (m ^ float64_p_values.view(np.int64))

        t = (new_values.view(float64_p_values.dtype) == np.nan) |\
            (np.abs(new_values.view(float64_p_values.dtype)) == np.inf)
        new_values[t] ^= np.random.randint(0, (1 << 11) - 2, size=np.sum(t), dtype=np.int64) << 52

        li, ls = interval
        new_values = new_values.view(float64_p_values.dtype)
        new_values = (new_values.clip(-1000, 1000)+1000)/2000 * (ls - li) + li
        values.scatter_nd_update(indices, new_values)


def mutate_weights_and_biases(pop, prob_w, prob_b, interval_w=None, interval_b=None,
                              mutate_uniform_w=None, mutate_uniform_b=None):
    for individual in pop:
        for layer in individual.model.layers:
            if type(layer) not in layers_available_for_evolution:
                continue

            weights = layer.trainable_weights[0]
            mutate(weights, prob_w, interval_w, mutate_uniform_w)

            biases = layer.trainable_weights[1]
            mutate(biases, prob_b, interval_b, mutate_uniform_b)


def mutate_replace(pop, prob_w, prob_b, interval_w=(-1, 1), interval_b=(-1, 1)):
    for individual in pop:
        for layer in individual.model.layers:
            if type(layer) not in layers_available_for_evolution:
                continue

            weights = layer.trainable_weights[0]
            indexes = np.argwhere(np.random.random(size=weights.shape) < prob_w)
            weights.scatter_nd_update(indexes, np.random.uniform(*interval_w, size=(len(indexes),)))

            biases = layer.trainable_weights[1]
            indexes = np.argwhere(np.random.random(size=biases.shape) < prob_b)
            biases.scatter_nd_update(indexes, np.random.uniform(*interval_b, size=(len(indexes),)))


def weights_and_biases(model):
    w = [None for i in range(len(model.layers)) if type(model.layers[i]) in layers_available_for_evolution]
    b = [None for _ in range(len(w))]

    i = 0
    for layer in model.layers:
        if type(layer) not in layers_available_for_evolution:
            continue
        w[i], b[i] = layer.trainable_weights
        i += 1

    return w, b


def cross(a, b, p):
    s1, s2 = a.shape, b.shape
    a, b = a.numpy().reshape((-1,)), b.numpy().reshape((-1,))
    a[:p], b[:p] = b[:p], a[:p].copy()
    return tf.Variable(a.reshape(s2)), tf.Variable(b.reshape(s1))


def crossover_2parents(model1, model2):
    global layers_format
    w1, b1 = weights_and_biases(model1)
    w2, b2 = weights_and_biases(model2)

    trainable_weights = 0
    trainable_biases = 0
    for w in w1:
        trainable_weights += np.prod(w.shape)
    for b in b1:
        trainable_biases += np.prod(b.shape)

    p = np.random.randint(1, trainable_weights)
    w_params = 0
    for i, w_val in enumerate(zip(w1, w2)):
        w_params += np.prod(w_val[0].shape)
        if w_params < p:
            w1[i], w2[i] = cross(*w_val, None)
        else:
            w1[i], w2[i] = cross(*w_val, w_params - p + 1)
            break

    p = np.random.randint(1, trainable_biases)
    b_params = 0
    for i, b_val in enumerate(zip(b1, b2)):
        b_params += np.prod(b_val[0].shape)
        if b_params < p:
            b1[i], b2[i] = cross(*b_val, None)
        else:
            b1[i], b2[i] = cross(*b_val, b_params - p + 1)
            break

    child1 = Individual(layers_format, loss, weights=w1, biases=b1)
    child2 = Individual(layers_format, loss, weights=w2, biases=b2)
    return child1, child2


def crossover(pop, prob):
    r = [(rand, i) for rand, i in zip(np.random.random(len(pop)), range(len(pop)))]
    r.sort(key=lambda k: k[0])
    i = 0
    pop_size = len(pop)
    while i < pop_size and r[i][0] < prob:
        if i+1 < pop_size and (r[i+1][0] < prob or np.random.random() < 0.5):
            child1, child2 = crossover_2parents(pop[r[i][1]].model, pop[r[i+1][1]].model)
            pop.append(child1)
            pop.append(child2)
        i += 2


def fitness(individual, features, labels):
    loss = individual.model.evaluate(features, labels, batch_size=5000)
    return 1/loss


def bin_search(val, v):
    i, bit = 0, 1
    while bit < len(v):
        bit <<= 1

    while bit:
        if (i | bit) < len(v) and v[i | bit] <= val:
            i |= bit
        bit >>= 1

    if v[i | bit] >= val:
        return i
    return i + 1


def selection(pop, pop_size, sample_features, sample_labels, elitism=0.0, fitness_interval=None):
    sf = np.zeros(len(pop))

    fitness_values = np.array([fitness(pop[i], sample_features, sample_labels) for i in range(len(pop))])
    if fitness_interval:
        a, b = fitness_interval
        abs_min_fitness = abs(np.min(fitness_values))
        fitness_values = (fitness_values + abs_min_fitness)/(np.max(fitness_values) + abs_min_fitness) * (b - a) + a

    sf[0] = fitness_values[0]
    for i in range(1, len(pop)):
        sf[i] = sf[i-1] + fitness_values[i]
    new_pop = [None for _ in range(pop_size)]

    start = 0
    if elitism:
        start = min(int(len(pop)*elitism), pop_size)
        x = [[sf[0], i] for i in range(len(pop))]
        for i in range(1, len(pop)):
            x[i][0] = sf[i]-sf[i-1]
        x.sort(key=lambda k: k[0], reverse=True)
        for i in range(start):
            new_pop[i] = pop[x[i][1]]

    for i in range(start, pop_size):
        pos = np.random.random()*sf[-1]
        new_pop[i] = pop[bin_search(pos, sf)]

    return new_pop


def best_individual(pop, features, labels):
    best = [0, fitness(pop[0], features, labels)]
    for i, individual in enumerate(pop[1:], 1):
        f = fitness(individual, features, labels)
        if f > best[1]:
            best[:] = i, f
    return pop[best[0]], best[1]


def learning_stage(pop, size_learning_group, features, labels, kwargs):
    r = np.random.choice(range(len(pop)), size_learning_group)
    for r_val in r:
        pop[r_val].model.fit(features, labels, **kwargs)


def genetic_algorithm(pop_size, mutate_prob_w, mutate_prob_b, mutate_replace_prob,
                      mutate_replace_interval_w, mutate_replace_interval_b,
                      crossover_prob, gen_max, size_learning_group, learning_params,
                      elitism=0.0, log_iter=10, fitness_interval=None,
                      mutate_interval_w=None, mutate_interval_b=None, mutate_uniform_w=True, mutate_uniform_b=True):
    log_file = open('logging.log', 'w')
    pop = [Individual(layers_format, loss) for _ in range(pop_size)]
    total_duration = 0
    for gen in range(gen_max):
        start_timestamp = time()

        mutate_weights_and_biases(pop, mutate_prob_w, mutate_prob_b, mutate_interval_w, mutate_interval_b,
                                  mutate_uniform_w, mutate_uniform_b)
        mutate_replace(pop, prob_w=mutate_replace_prob[0], prob_b=mutate_replace_prob[1],
                       interval_w=mutate_replace_interval_w, interval_b=mutate_replace_interval_b)
        crossover(pop, crossover_prob)
        learning_stage(pop, size_learning_group, train_feature, train_label, learning_params)
        pop = selection(pop, pop_size, train_feature, train_label, elitism, fitness_interval)

        if gen % log_iter == log_iter - 1:
            best = best_individual(pop, train_feature, train_label)
            duration = time() - start_timestamp
            print(f'Generation {gen}: best individual fitness = {best[1]}. Duration: {duration}',
                  file=log_file, flush=True if gen % 10 == 0 else False)
            total_duration += duration
    best = best_individual(pop, train_feature, train_label)
    print('Best individual fitness:', best[1], 'Loss:', 1/best[1], 'Total duration:', total_duration, file=log_file)
    log_file.close()
    best[0].model.save('best_neural_network')
    return best[0]


def main():
    winner_net = genetic_algorithm(gen_max=300, pop_size=50, mutate_prob_w=0.0005, mutate_prob_b=0.0005,
                                   mutate_interval_w=(-3, 3), mutate_interval_b=(-0.7, 0.7),
                                   mutate_uniform_w=None, mutate_uniform_b=None,
                                   mutate_replace_prob=[0.0005, 0.0001],
                                   mutate_replace_interval_w=(-1, 1), mutate_replace_interval_b=(-0.7, 0.7),
                                   crossover_prob=0.8, elitism=3,
                                   size_learning_group=2, learning_params={'epochs': 2, 'batch_size': 200},
                                   log_iter=1).model  # fitness_interval=(0.1, 10)

    loss = winner_net.evaluate(train_feature, train_label, batch_size=2500)
    print('train_loss:', loss)
    loss = winner_net.evaluate(test_feature, test_label, batch_size=2500)
    print('test_loss:', loss)

    show_plot(winner_net.predict(test_feature[:2500]), test_label[:2500],
              winner_net.predict(train_feature[:5000]), train_label[:5000],
              (0.5, 0.5), 600)


if __name__ == '__main__':
    main()
