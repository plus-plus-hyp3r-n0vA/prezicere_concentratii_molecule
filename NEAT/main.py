import pandas as pd
import numpy as np

import neat
import pickle
import random
from multiprocessing import Pool

feature = pd.read_csv('../data/feature_nirspec100.csv').values
label = pd.read_csv('../data/label_nirspec100.csv').values

# 100 random data
# p = [4826, 31757, 25834, 26607, 37618, 16953, 1633, 38559, 16174, 6779, 28881, 14980, 38729, 13816, 32933, 7992, 39395, 3649, 9174, 39568, 18124, 21122, 7959, 15201, 31648, 39339, 29350, 37429, 13912, 24092, 10823, 32187, 12371, 25361, 5640, 25517, 38293, 18361, 2025, 24981, 34470, 2797, 5242, 33505, 25206, 23115, 11510, 28940, 16147, 14488, 6678, 18509, 16665, 30961, 31126, 29867, 39406, 1107, 19318, 29004, 15840, 36030, 34252, 16743, 17627, 8950, 25917, 31001, 299, 8428, 39252, 28270, 17296, 7729, 14925, 345, 23389, 23128, 14290, 37979, 21191, 21253, 13476, 9408, 16741, 26915, 34880, 16170, 38485, 38943, 13480, 20817, 23309, 19511, 9810, 34506, 2270, 37415, 206, 19076]

# 100 best data
# p = [7, 17, 21, 48, 54, 60, 66, 74, 88, 90, 95, 110, 111, 112, 118, 140, 160, 170, 171, 174, 197, 202, 206, 217, 220, 241, 248, 260, 274, 287, 307, 311, 314, 316, 322, 328, 341, 353, 355, 361, 365, 373, 376, 384, 385, 386, 389, 390, 393, 398, 403, 406, 418, 422, 423, 427, 433, 434, 451, 452, 484, 487, 499, 509, 511, 516, 531, 538, 543, 554, 578, 588, 594, 597, 610, 614, 615, 618, 629, 642, 652, 669, 677, 678, 688, 692, 700, 705, 709, 717, 722, 725, 731, 734, 740, 745, 757, 759, 760, 775]

# 500 random data
# p = [39471, 21271, 6166, 15539, 16547, 8663, 18781, 25804, 36849, 13402, 1795, 37280, 36169, 8283, 3070, 10289, 30367, 16132, 28336, 2640, 27262, 17825, 16106, 26928, 22123, 24854, 20468, 28205, 7443, 13538, 20631, 20336, 24096, 39430, 1285, 1641, 28540, 13874, 10780, 13543, 13455, 34294, 21867, 7158, 2786, 22231, 26884, 894, 14476, 10834, 22398, 33811, 26460, 31587, 5058, 1781, 19564, 7533, 30510, 18127, 22994, 8435, 39374, 37616, 13752, 18108, 16186, 22259, 24150, 22764, 20181, 13258, 32398, 7565, 3687, 4075, 22297, 31594, 5972, 31666, 21992, 29319, 36982, 24373, 26532, 11458, 13556, 22194, 21381, 1006, 2997, 96, 2116, 10810, 21511, 22985, 248, 15652, 7292, 21184, 36710, 22779, 10103, 30513, 36579, 1632, 7909, 24980, 11970, 17495, 1630, 8280, 3846, 31535, 1123, 29879, 22178, 25833, 877, 22502, 9918, 33725, 1778, 3815, 16926, 12689, 39558, 23180, 288, 1309, 26005, 17782, 19225, 32509, 30621, 35261, 17880, 7651, 901, 39381, 27042, 23679, 25586, 30550, 19782, 31386, 36708, 6489, 14460, 3976, 33924, 8309, 14627, 10762, 19630, 30680, 28573, 31294, 23547, 33259, 21514, 29362, 17791, 13016, 11231, 10862, 6404, 32935, 15542, 29873, 35417, 19404, 6997, 9664, 6142, 8163, 22907, 17703, 30423, 23816, 13438, 28962, 24760, 1516, 35666, 20136, 12903, 450, 35821, 17826, 20398, 22488, 7232, 4991, 10755, 25925, 27185, 9310, 32594, 24213, 8343, 27063, 33919, 24394, 27387, 15757, 34946, 39969, 24926, 9615, 21612, 32885, 31646, 34132, 29962, 27899, 7509, 12056, 5729, 37187, 14169, 27218, 37998, 19141, 25702, 15041, 12639, 14866, 6889, 12963, 29776, 8632, 12630, 1464, 17231, 17025, 13629, 36454, 12252, 32956, 12079, 391, 8441, 21464, 3458, 35562, 38915, 6309, 28721, 22943, 37101, 15325, 27452, 15731, 1445, 29199, 1455, 39190, 6046, 30546, 30753, 9348, 11167, 16287, 30454, 5110, 37306, 35581, 2774, 19155, 32431, 32637, 21769, 38342, 31217, 21105, 30242, 7087, 38815, 688, 29919, 5316, 6149, 31803, 4709, 6280, 11802, 19910, 22257, 6399, 20162, 13676, 14914, 37993, 19852, 11693, 34845, 18428, 10442, 13370, 5609, 37416, 32328, 6038, 7934, 33925, 38146, 31416, 26358, 2355, 35140, 33291, 29593, 12580, 12804, 18266, 27569, 37409, 1071, 32123, 29284, 5083, 29725, 18688, 34397, 32619, 33487, 26656, 5696, 32362, 22791, 27758, 14188, 26747, 31603, 31993, 32090, 32655, 1720, 32314, 29916, 4686, 9769, 29662, 38523, 19235, 25196, 35640, 13201, 8792, 7272, 18475, 39367, 16010, 33672, 13376, 1041, 20984, 30498, 25503, 31065, 15024, 6393, 38008, 1431, 21504, 6419, 2096, 2148, 15330, 38685, 14979, 32893, 13531, 38090, 11604, 18806, 13385, 19030, 37425, 20041, 18699, 1096, 30346, 29724, 38947, 2952, 39198, 39530, 31185, 19328, 8639, 25729, 2680, 28421, 23219, 19012, 21210, 13591, 24869, 5777, 9580, 10322, 16600, 22309, 31302, 2022, 13993, 8382, 37919, 28885, 7401, 8372, 24758, 9343, 34771, 29520, 21654, 27672, 34069, 17846, 33728, 26213, 28639, 22207, 27364, 6694, 31675, 11193, 36501, 25218, 10477, 4226, 28534, 38753, 24347, 10765, 13921, 13159, 17080, 20289, 31157, 1761, 16099, 38826, 34966, 36408, 244, 7168, 22893, 2868, 12521, 6080, 13775, 383, 12983, 27076, 8180, 35810, 148, 7462, 12607, 28979, 35653, 33726, 14661, 10679, 29650, 33961, 35244, 7843, 5062, 22554, 15687, 6363, 17492, 21338, 20972, 36888, 23306, 18354, 2973, 1773, 6464, 38998, 37812, 5453, 5132, 30251, 30181, 28189, 38780, 18623, 6269, 38783, 36057, 18194, 23987, 18309, 37804]

# 500 best data
p = [7, 17, 21, 32, 35, 47, 48, 50, 54, 64, 65, 74, 90, 95, 109, 110, 111, 112, 118, 140, 160, 170, 171, 174, 181, 189, 197, 199, 202, 206, 207, 217, 219, 224, 229, 241, 247, 258, 260, 262, 273, 274, 307, 310, 311, 316, 327, 328, 341, 353, 365, 373, 376, 384, 385, 389, 393, 400, 406, 418, 422, 423, 434, 447, 452, 466, 484, 485, 486, 487, 496, 499, 517, 528, 531, 535, 538, 543, 554, 565, 578, 582, 588, 594, 595, 597, 610, 615, 618, 629, 642, 652, 663, 669, 677, 678, 692, 700, 705, 734, 745, 751, 757, 759, 761, 775, 785, 791, 793, 794, 805, 807, 808, 825, 832, 860, 873, 876, 881, 883, 884, 903, 906, 910, 911, 912, 914, 926, 932, 941, 946, 954, 955, 956, 957, 966, 974, 978, 979, 983, 991, 1001, 1003, 1005, 1006, 1011, 1016, 1033, 1044, 1048, 1052, 1057, 1059, 1062, 1065, 1071, 1073, 1078, 1079, 1085, 1096, 1103, 1106, 1108, 1115, 1124, 1125, 1131, 1137, 1150, 1160, 1161, 1173, 1187, 1188, 1198, 1202, 1222, 1224, 1254, 1262, 1263, 1264, 1276, 1282, 1283, 1300, 1305, 1306, 1311, 1327, 1334, 1335, 1338, 1356, 1358, 1361, 1387, 1395, 1399, 1401, 1413, 1414, 1419, 1428, 1434, 1443, 1446, 1454, 1460, 1476, 1488, 1512, 1513, 1526, 1527, 1531, 1545, 1557, 1558, 1559, 1584, 1607, 1621, 1627, 1641, 1648, 1665, 1667, 1685, 1692, 1699, 1707, 1708, 1715, 1719, 1721, 1746, 1755, 1793, 1800, 1807, 1818, 1823, 1827, 1840, 1843, 1850, 1883, 1887, 1890, 1892, 1915, 1930, 1931, 1933, 1938, 1941, 1955, 1956, 1959, 1962, 1963, 1974, 1977, 1984, 1987, 1988, 2003, 2004, 2009, 2021, 2031, 2040, 2041, 2046, 2047, 2050, 2054, 2056, 2065, 2068, 2079, 2090, 2095, 2096, 2102, 2109, 2116, 2119, 2126, 2168, 2171, 2182, 2183, 2187, 2197, 2246, 2286, 2302, 2303, 2306, 2315, 2318, 2326, 2330, 2345, 2346, 2362, 2371, 2374, 2380, 2381, 2384, 2388, 2399, 2402, 2408, 2419, 2429, 2443, 2456, 2462, 2464, 2466, 2469, 2470, 2474, 2492, 2498, 2500, 2505, 2506, 2507, 2508, 2509, 2527, 2545, 2552, 2561, 2572, 2573, 2578, 2581, 2595, 2599, 2600, 2606, 2608, 2620, 2626, 2643, 2646, 2647, 2648, 2657, 2668, 2674, 2704, 2708, 2710, 2722, 2725, 2735, 2744, 2745, 2756, 2762, 2787, 2802, 2829, 2832, 2835, 2869, 2878, 2879, 2880, 2907, 2921, 2943, 2951, 2952, 2958, 2963, 2965, 2966, 2970, 2987, 2989, 2994, 2998, 3005, 3012, 3019, 3020, 3031, 3042, 3049, 3060, 3066, 3067, 3073, 3081, 3082, 3094, 3095, 3101, 3106, 3139, 3147, 3155, 3156, 3167, 3174, 3184, 3185, 3189, 3190, 3214, 3216, 3218, 3248, 3268, 3272, 3281, 3282, 3291, 3292, 3302, 3327, 3330, 3332, 3337, 3339, 3343, 3358, 3359, 3366, 3369, 3388, 3389, 3391, 3392, 3394, 3403, 3404, 3408, 3414, 3416, 3428, 3437, 3441, 3442, 3446, 3460, 3475, 3485, 3509, 3511, 3513, 3523, 3532, 3539, 3542, 3543, 3561, 3562, 3579, 3584, 3594, 3602, 3603, 3605, 3617, 3622, 3625, 3635, 3649, 3654, 3663, 3702, 3714, 3716, 3727, 3731, 3738, 3753, 3756, 3758, 3762, 3773, 3779, 3785, 3787, 3790, 3801, 3805, 3814, 3817, 3820]


train_feature = feature[p]
train_label = label[p]
test_feature = feature[50000:]
test_label = label[50000:]
features_and_labels = list(zip(train_feature, train_label))


def change_config(filename, data):
    new_config = ""
    with open(filename) as fd:
        for row in fd:
            k = row.split('=', 1)[0].strip()
            if k in data:
                new_config += f'{k} = {data[k]}\n'
            else:
                new_config += row

    with open(filename, 'w') as fd:
        print(new_config, file=fd)


def new_pop_with_config(config, population):
    genome_indexer = population.reproduction.genome_indexer
    ancestors = population.reproduction.ancestors
    node_indexer = population.config.genome_config.node_indexer

    new_pop = neat.Population(config, (population.population, population.species, population.generation))
    for reporter in population.reporters.reporters:
        new_pop.add_reporter(reporter)

    new_pop.config.genome_config.node_indexer = node_indexer
    new_pop.reproduction.genome_indexer = genome_indexer
    new_pop.reproduction.ancestors = ancestors
    return new_pop


def fitness(expected_out, predicted_out, mse=False):
    mean_squared_error = np.mean(np.square(expected_out - predicted_out))

    return mean_squared_error if mse else 1/mean_squared_error


def eval_genome(genome, config, inp, out):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    outputs = np.zeros((len(inp), 12))

    for i, si in enumerate(inp):
        outputs[i] = net.activate(si)

    return fitness(out, outputs)


def eval_genomes(genomes, config):
    sample_input, sample_outputs = list(zip(*random.sample(features_and_labels, 100)))

    jobs = []
    for genome_id, genome in genomes:
        jobs.append(pool.apply_async(eval_genome, (genome, config, sample_input, sample_outputs)))

    for job, (genome_id, genome) in zip(jobs, genomes):
        genome.fitness = job.get()


def run(config_file, changes=(), check_frequency=50, condition=lambda old_f, new_f: 0 <= new_f-old_f < 0.01):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    if changes:
        no_generations = sum([i for i, c in changes])
        last_config_iter = 0

        change_config(config_file, changes[0][1])
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        p.add_reporter(neat.Checkpointer(no_generations/5, 18000))
        winner = p.run(eval_genomes, 1)
        last_winner = neat.nn.FeedForwardNetwork.create(winner, config)
        it = 1
        for generation in range(1, no_generations):
            winner = p.run(eval_genomes, 1)

            cond = condition(fitness(train_label, winner_activate(last_winner, train_feature)),
                             fitness(train_label, winner_activate(neat.nn.FeedForwardNetwork.create(winner, config),
                                                                  train_feature)))
            if generation % check_frequency == 0:
                last_winner = neat.nn.FeedForwardNetwork.create(winner, config)
            if len(changes) >= last_config_iter + 1 and \
                    ((check_frequency > 0 and generation % check_frequency == 0 and cond) or
                     changes[last_config_iter][0] <= it):
                it = 0
                last_config_iter += 1

                if len(changes) == last_config_iter:
                    break
                with open('config_changes.txt', 'a') as fd:
                    print('NEW CONFIG', last_config_iter, 'GENERATION', generation, file=fd)

                change_config(config_file, changes[last_config_iter][1])
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_file)
                p = new_pop_with_config(config, p)
            it += 1
    else:
        p.add_reporter(neat.Checkpointer(100, 18000))
        winner = p.run(eval_genomes, 500)

    pool.close()
    pool.join()

    with open('output_best_genome.txt', 'w') as fd:
        print('\nBest genome:\n{!s}'.format(winner), file=fd)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open(f'winner_net', 'wb') as f:
        pickle.dump(winner_net, f)

    return winner_net


def winner_activate(winner_net, features):
    outputs = np.zeros((len(features), 12))
    for i, si in enumerate(features):
        outputs[i] = np.array(winner_net.activate(si))
    return outputs


if __name__ == '__main__':
    configs = ((1000, {'weight_mutate_power': 0.005, 'weight_mutate_rate': 0.65,
                       'conn_add_prob': 0.6, 'conn_delete_prob': 0.5}),
               (1000, {'weight_mutate_power': 0.0005, 'weight_mutate_rate': 0.85}),
               (500, {'weight_mutate_power': 0.00025, 'weight_mutate_rate': 0.45,
                      'conn_add_prob': 0.1, 'conn_delete_prob': 0.1}),
               )
    # configs = ((2500, {'weight_mutate_power': 0.0005, 'weight_mutate_rate': 0.85,
    #                    'conn_add_prob': 0.6, 'conn_delete_prob': 0.5}),
    #            )
    pool = Pool(6)

    config_path = 'config-feedforward.txt'
    winner_net = run(config_path, configs)

    outputs = winner_activate(winner_net, train_feature)

    print('Error train set:', fitness(train_label, outputs), fitness(train_label, outputs, True))

    outputs = winner_activate(winner_net, test_feature)

    print('Error test set:', fitness(test_label, outputs), fitness(test_label, outputs, True))
