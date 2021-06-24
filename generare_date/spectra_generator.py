import os

from CONSTANTS_AND_RANGES_PSG import *
import random
import subprocess
import numpy as np
from datetime import datetime
import astropy.constants as const
from PT import PT_line


def dict_to_psg_file_str(d, prefix, exception=()):
    s = ''
    if type(d) is not dict:
        return ''
    for key, val in d.items():
        if key not in exception:
            s += f'<{prefix.upper()}-{key.upper().replace("_", "-")}>{val}\n'
    return s


def generate_object_params():
    index_star_type = random.randint(3, len(LIST_STAR_TYPE) - 1)
    sma = random.uniform(0.5, 8) if index_star_type > 3 else random.uniform(5, 25)
    Tstar = random.uniform(*RANGE_STAR_TEMPERATURE_BY_STAR_TYPE[index_star_type])
    Rstar = random.uniform(*RANGE_STAR_RADIUS_BY_STAR_TYPE[index_star_type])
    Lstar = None
    canholdatmo = False

    while not canholdatmo:
        r = random.random()
        rplanet = earths_radius = random.uniform(0.3, 4) if r < 0.7 else random.uniform(6, 12)
        mplanet = earths_mass = random.uniform(0.01, 3) if r < 0.7 else random.uniform(3, 400)
        gravity = 9.80665 * earths_mass / earths_radius ** 2  # g ~ m/r^2 (/wiki/Surface_gravity)

        # code taken from INARA
        Lstar = const.sigma_sb.value * 4. * np.pi * (Rstar * const.R_sun.value) ** 2 * Tstar ** 4

        vesc = (2. * const.G.value * mplanet * const.M_earth.value / (rplanet * const.R_earth.value)) ** 0.5 / 1000.
        Insol = Lstar / const.L_sun.value / sma ** 2
        shoreline = np.log10(1e4 / 1e-6) / np.log10(70. / 0.2)

        pinsol = 1e4 * (vesc / 70.) ** shoreline
        if Insol < pinsol:
            canholdatmo = True
        ########################

    if LIST_STAR_TYPE[index_star_type] == 'M':
        M_star = (Lstar / (const.L_sun.value * 0.23)) ** (1 / 2.3)  # Calculate stellar mass [solar masses]
    else:
        M_star = (Lstar / const.L_sun.value) ** (1 / 4)
    period = np.sqrt(sma**3 / M_star) * EARTH_ORBITAL_PERIOD

    return {
        'date': datetime.utcnow().strftime('%Y/%m/%d %H:%M'),
        'diameter': earths_radius * 6371 * 2,
        'gravity': gravity,
        'gravity_unit': 'g',  # m/s^2
        'star_distance': sma,
        'star_velocity': random.uniform(-60, -3) if random.random() < 0.5 else random.uniform(3, 60),
        'solar_longitude': random.randint(*RANGE_OBJECT_SOLAR_LONGITUDE),
        'solar_latitude': random.randint(*RANGE_OBJECT_SOLAR_LATITUDE),
        'season': 180,
        'star_type': LIST_STAR_TYPE[index_star_type],
        'star_temperature': Tstar,
        'star_radius': Rstar,
        'obs_longitude': random.uniform(*RANGE_OBJECT_OBS_LONGITUDE),
        'obs_latitude': random.uniform(*RANGE_OBJECT_OBS_LATITUDE),
        'obs_velocity': random.uniform(-300, 300),
        'period': period,
        'star_metallicity': random.uniform(-1, 1),
        'inclination': random.uniform(88.5, 91.5),
        'eccentricity': random.randint(*RANGE_OBJECT_ECCENTRICITY),
        'periapsis': random.randint(*RANGE_OBJECT_PERIAPSIS)
    }


def generate_geometry_params():
    geometry = {
        'geometry': 'Observatory',
        'obs_altitude': random.uniform(0.1, 500),
        'altitude_unit': 'pc',
        'disk_angles': 1
    }
    return geometry


def generate_layers(nr_layers, abundance, obj):
    kappa = np.random.uniform(-3.5, -2.0)
    gamma1 = np.random.uniform(-1.5, 1.1)
    gamma2 = np.random.uniform(-1.5, 0.)
    alpha = np.random.uniform(0., 1.)
    beta = np.random.uniform(0.7, 1.1)
    layers = []
    pressure = abs(np.random.normal(1, 1.5))
    if pressure < 1e-3:
        pressure = np.random.uniform(1e-3, 1e-1)

    min_pressure = np.random.uniform(1e-10, pressure)
    pressure_profile = np.logspace(np.log(pressure), np.log(min_pressure), nr_layers, base=np.e)

    temp = PT_line(pressure_profile, [kappa, gamma1, gamma2, alpha, beta], obj['star_radius'] * const.R_sun.value,
                   obj['star_temperature'], 0, obj['star_distance'] * const.au.value, obj['gravity']*100)

    for index_layer in range(nr_layers):
        layers.append(','.join([str(pressure_profile[index_layer]), str(temp[index_layer]), *abundance]))

    return temp[0], pressure, layers


def generate_atmosphere_params(ngas, obj):
    molecules_arr = np.array(list(CHOSEN_MOLECULES.keys()))
    np.random.shuffle(molecules_arr)

    random_abundance_float = np.random.dirichlet(np.ones(ngas))
    random_abundance_str = [str(val) for val in random_abundance_float]
    weight = np.sum(random_abundance_float * CHOSEN_MOLECULES_WEIGHTS)

    temperature, pressure, layers = generate_layers(np.random.randint(25, 35), random_abundance_str, obj)

    return {
        'structure': 'Equilibrium',
        'punit': 'bar',
        'pressure': pressure,
        'weight': weight,
        'temperature': temperature,
        'ngas': ngas,
        'gas': ','.join(molecules_arr[:ngas]),
        'type': ','.join([CHOSEN_MOLECULES[i] for i in molecules_arr[:ngas]]),
        'abun':  ('1,' * ngas)[:-1],
        'unit': ('scl,' * ngas)[:-1],
        'naero': 0,
        'nmax': 0,
        'lmax': 0,
        'layers': layers
    }


def generate_surface_params():
    albedo = random.uniform(0.05, 0.8)
    return {
        'albedo': albedo,
        'emissivity': 1 - albedo,  # albedo + emissivity = 1 for non transparent bodies
        'gas_ratio': 0,
        'nsurf': random.randint(*RANGE_SURFACE_NSURF),
        'phaseg': random.randint(RANGE_SURFACE_PHASEG[0], RANGE_SURFACE_PHASEG[1]+1),
        'gas_unit': 'ratio'
    }


def generate_config_file(file_name, obj, geometry=None, atmosphere=None, surface=None, generator=None):
    file_content = '''<OBJECT>Exoplanet
<OBJECT-NAME>Random exoplanet
''' + dict_to_psg_file_str(obj, 'object') + \
      dict_to_psg_file_str(geometry, 'geometry') + \
      dict_to_psg_file_str(atmosphere, 'atmosphere', ('layers', 'temperature')) + \
      f'''<ATMOSPHERE-LAYERS-MOLECULES>{atmosphere['gas']}
<ATMOSPHERE-LAYERS>{len(atmosphere['layers'])}
<SURFACE-TEMPERATURE>{atmosphere['temperature']}
'''
    for i, layer in enumerate(atmosphere['layers'], 1):
        file_content += f'<ATMOSPHERE-LAYER-{i}>{layer}\n'
    file_content += dict_to_psg_file_str(surface, 'surface') + \
        dict_to_psg_file_str(generator, 'generator')

    os.makedirs('../data/R100/config', exist_ok=True)
    with open('../data/R100/config/' + file_name, 'w') as fd:
        fd.write(file_content)


def generate_spectrum(file_name, api='http://localhost:3000/api.php'):
    # psg site: https://psg.gsfc.nasa.gov/api.php

    os.makedirs('../data/R100/spectrum', exist_ok=True)
    path_config = f'../data/R100/config/{file_name}'
    path_spectrum = f'../data/R100/spectrum/{file_name}'

    comm = f'curl -s -d type=trn --data-urlencode file@{path_config} {api} -o {path_spectrum}'
    with subprocess.Popen(comm.split(), stdout=subprocess.PIPE, shell=False) as proc:
        output = proc.stdout.read().decode("utf-8")
        print(output)


if __name__ == '__main__':
    for i in range(0, 1000):
        file_name = f'file{i}.txt'
        obj = generate_object_params()
        generate_config_file(file_name, obj, generate_geometry_params(),
                             generate_atmosphere_params(len(CHOSEN_MOLECULES), obj), generate_surface_params(),
                             JWST_NIRSPEC_PRISM_TELESCOPE_PARAM)
        generate_spectrum(file_name)
