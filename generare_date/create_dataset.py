import os
from freq_map.freq_to_index_nirspec100 import freq_to_index
from freq_map.index_to_freq_nirspec100 import freq
import math
import numpy as np


bad_spectra = []
ROOT_CONFIG = '../data/R100/config'
ROOT_SPECTRUM = '../data/R100/spectrum'


def find_abundance_values(filename):
    path = os.path.join(ROOT_CONFIG, filename)
    tag = '<ATMOSPHERE-LAYER-1>'
    with open(path, 'r') as fd:
        for row in fd:
            if row.startswith(tag):
                values = row[len(tag):-1].split(',')
                return values[2:]
    return None


def extract_data_from_file(filepath, file):
    global bad_spectra
    molecules = None
    molecular_abundances = find_abundance_values(file)
    freq_data = [0.0 for _ in range(len(freq))]
    with open(filepath, 'r') as fd:
        for row in fd:
            comment_molecules = '# Molecules considered: '
            if comment_molecules in row:
                molecules = row[len(comment_molecules):-1].split(',')
            if row[0] == '#':
                continue
            freq_p, freq_i = row.split()[:2]
            float_freq_i = float(freq_i)
            if math.isnan(float_freq_i) or float_freq_i < 0:
                print('invalid frequency in file:', filepath)
                bad_spectra.append(filepath)
                return None
            else:
                freq_data[freq_to_index[freq_p]] = str(abs(float_freq_i))
    return molecules, molecular_abundances, freq_data


def create_dataset():
    header = True
    with open('../data/feature_nirspec100.csv', 'w') as feature_fd,\
            open('../data/label_nirspec100.csv', 'w') as label_fd:
        for file in os.listdir(ROOT_SPECTRUM):
            if file.startswith('file') and os.stat(os.path.join(ROOT_SPECTRUM, file)).st_size > 1000:
                try:
                    data = extract_data_from_file(os.path.join(ROOT_SPECTRUM, file), file)
                    molecules, molecular_abundances = zip(*sorted(zip(*data[:2]), key=lambda x: x[0]))
                    if np.sum(np.fromiter(map(float, data[2]), dtype=float) == 0.0) > 1 * len(data[2]):
                        bad_spectra.append(file)
                        continue
                    if header:
                        print(','.join([f'f{i}' for i in range(len(data[2]))]), file=feature_fd)
                        print(','.join(molecules), file=label_fd)
                        header = False
                    print(','.join(data[2]), file=feature_fd)
                    print(','.join(molecular_abundances), file=label_fd)
                except Exception as e:
                    print('Exception: ', e)


if __name__ == '__main__':
    create_dataset()
    with open('bad', 'w') as fd:
        print(bad_spectra, file=fd)

