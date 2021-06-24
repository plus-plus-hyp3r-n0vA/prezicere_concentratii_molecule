if __name__ == '__main__':
    freqs = []

    for i in range(1):  # >1 daca ar trebui generat un set de date cu spectre de rezolutii din mai multe regiuni
        filename = f'spectra_examples/nirspec100_file{i}.txt'
        with open(filename, 'r') as fd:
            for line in fd:
                if line.startswith('#'):
                    continue
                toks = line.split(' ')
                freqs.append(toks[0])

    freqs = list(set(freqs))
    freqs.sort(key=lambda x: float(x))

    with open('freq_map/index_to_freq_nirspec100.py', 'w') as fd:
        fd.write('freq = ' + str(freqs))
    freq_map = {freqs[x]: x for x in range(len(freqs))}
    with open('freq_map/freq_to_index_nirspec100.py', 'w') as fd:
        fd.write('freq_to_index = ' + str(freq_map))
