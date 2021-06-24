RANGE_OBJECT_DIAMETER = (1e-2, 1e8)            # Diameter of the object [km]
RANGE_OBJECT_GRAVITY = (0, 1e35)               # Gravity/density/mass of the object
RANGE_OBJECT_STAR_DISTANCE = (0, 1e5)          # Distance of the planet to the Sun [AU], and for exoplanets the semi-major axis [AU]
RANGE_OBJECT_STAR_VELOCITY = (-1e4, 1e4)       # Velocity of the planet to the Sun [km/s], and for exoplanets the RV amplitude [km/s]
RANGE_OBJECT_SOLAR_LONGITUDE = (-360, 360)     # Sub-solar east longitude [degrees]
RANGE_OBJECT_SOLAR_LATITUDE = (-90, 90)        # Sub-solar latitude [degrees]
RANGE_OBJECT_SEASON = (0, 360)                 # Angular parameter (season/phase) that defines the position of the planet moving along its Keplerian orbit. For exoplanets, 0:Secondary transit, 180:Primary transit, 90/270:Opposition. For solar-system bodies, 0:'N spring equinox', 90:'N summer solstice', 180:'N autumn equinox', 270:'N winter solstice' [degrees]
RANGE_OBJECT_INCLINATION = (0, 90)             # Orbital inclination [degree], mainly relevant for exoplanets. Zero is phase on, 90 is a transiting orbit
RANGE_OBJECT_ECCENTRICITY = (0, 1)             # Orbital eccentricity, mainly relevant for exoplanets
RANGE_OBJECT_PERIAPSIS = (0, 360)              # Orbital longitude of periapse [degrees]. It indicates the phase at which the planet reaches periapsis
RANGE_OBJECT_STAR_TEMPERATURE = (1, 1e5)       # Temperature of the parent star [K]
RANGE_OBJECT_STAR_RADIUS = (1e-3, 1e8)         # Radius of the parent star [Rsun]
RANGE_OBJECT_STAR_METALLICITY = (-10.0, 10.0)  # Metallicity of the parent star and object with respect to the Sun in log [dex]
RANGE_OBJECT_OBS_LONGITUDE = (-360, 360)       # Sub-observer east longitude [degrees]
RANGE_OBJECT_OBS_LATITUDE = (-90, 90)          # Sub-observer latitude, for exoplanets inclination [degrees]
RANGE_OBJECT_OBS_VELOCITY = (-1e4, 1e4)        # Relative velocity between the observer and the object [km/s]
RANGE_OBJECT_PERIOD = (0, 1e8)                 # This field is computed by the geometry module - It is the apparent rotational period of the object as seen from the observer [days]


RANGE_GEOMETRY_OFFSET_NS = (-1e6, 1e6)         # Vertical offset with respect to the sub-observer location
RANGE_GEOMETRY_OFFSET_EW = (-1e6, 1e6)         # Horizontal offset with respect to the sub-observer location
RANGE_GEOMETRY_OBS_ALTITUDE = (0, 1e8)         # Distance between the observer and the surface of the planet
RANGE_GEOMETRY_AZIMUTH = (0, 360)              # The azimuth angle between the observational projected vector and the solar vector on the reference plane.
RANGE_GEOMETRY_USER_PARAM = (0, 1000)          # Parameter for the selected geometry, for Nadir / Lookingup this field indicates the zenith angle [degrees], for limb / occultations this field indicates the atmospheric height [km] being sampled
RANGE_GEOMETRY_STELLAR_TEMPERATURE = (1, 1e5)  # For stellar occultations, this field indicates the temperature [K] of the occultation star
RANGE_GEOMETRY_STELLAR_MAGNITUDE = (-30, 30)   # For stellar occultations, this field indicates the brightness [magnitude] of the occultation star
RANGE_GEOMETRY_DISK_ANGLES = (1, 10)           # This field allows to divide the observable disk in finite rings so radiative-transfer calculations are performed with higher accuracy
RANGE_GEOMETRY_PHASE = (0, 360)                # This field is computed by the geometry module - It indicates the phase between the Sun and observer
RANGE_GEOMETRY_STAR_FRACTION = (0, 1)          # This field is computed by the geometry module - It indicates how much the beam fills the parent star (1:maximum)
RANGE_GEOMETRY_STAR_DISTANCE = (-1, 1e8)       # This field is computed by the geometry module - It indicates the projected distance between the beam and the parent star in arcsceconds


RANGE_ATMOSPHERE_PRESSURE = (0, 1e35)          # For equilibrium atmospheres, this field defines the surface pressure; while for cometary coma, this field indicates the gas production rate
RANGE_ATMOSPHERE_TEMPERATURE = (1, 1e4)        # For atmospheres without a defined P/T profile, this field indicates the temperature across all altitudes
RANGE_ATMOSPHERE_WEIGHT = (20, 2e3)             # Molecular weight of the atmosphere [g/mol] or expansion velocity [m/s] for expanding atmospheres
RANGE_ATMOSPHERE_NGAS = (0, 20)                # Number of gases to include in the simulation
RANGE_ATMOSPHERE_NAERO = (0, 20)               # Number of aerosols to include in the simulation
RANGE_ATMOSPHERE_NMAX = (0, 100)               # When performing scattering aerosols calculations, this parameter indicates the number of n-stream pairs - Use 0 for extinction calculations only (e.g. transit, occultation)
RANGE_ATMOSPHERE_LMAX = (0, 100)               # When performing scattering aerosols calculations, this parameter indicates the number of scattering Legendre polynomials used for describing the phase function - Use 0 for extinction calculations only (e.g. transit, occultation)
RANGE_ATMOSPHERE_LAYERS = (0, 999)             # Number of layers of the atmospheric vertical profile


RANGE_SURFACE_TEMPERATURE = (1, 1e5)           # Temperature of the surface [K]
RANGE_SURFACE_ALBEDO = (0, 1.0)                # Albedo the surface [0:non-reflectance, 1:fully-reflective]
RANGE_SURFACE_EMISSIVITY = (0, 1.0)            # Emissivity of the surface [0:non-emitting, 1:perfect-emitter]
RANGE_SURFACE_PHASEG = (-1.0, 1.0)             # One-term Henyey-Greenstein g-factor [0:isotropic, -1:backward scatterer, +1:forward scatterer]
RANGE_SURFACE_GAS_RATIO = (0, 1e3)             # For expanding cometary coma, this value indicates an scaling value for the dust in the coma
RANGE_SURFACE_NSURF = (0, 20)                  # Number of components describing the surface properties [areal mixing]


RANGE_GENERATOR_RANGE1 = (1e-5, 1e7)           # Lower spectral range for the simulation
RANGE_GENERATOR_RANGE2 = (1e-5, 1e7)           # Upper spectral range for the simulation
RANGE_GENERATOR_RESOLUTION = (1e-6, 1e8)       # Spectral resolution for the simulation. PSG assumes that the sampling resolution is equal is to the instrumental resolution, yet radiative transfer resolutions are always performed at the necessary/higher resolutions in order to accurately describe the lineshapes
RANGE_GENERATOR_BEAM = (1e-3, 1e6)             # Full width half-maximum (FWHM) of the instrument's beam or field-of-view (FOV)
RANGE_GENERATOR_DIAMTELE = (1e-5, 1e5)         # Diameter of the main reflecting surface of the telescope or instrument [m]
RANGE_GENERATOR_NOISETIME = (0.0, 1e7)         # Exposure time per frame [sec]
RANGE_GENERATOR_NOISEFRAMES = (1, 1e9)         # Number of exposures
RANGE_GENERATOR_NOISEPIXELS = (1, 1e9)         # Total number of pixels that encompass the beam (GENERATOR-BEAM) and the spectral unit (GENERATOR-RESOLUTION)
RANGE_GENERATOR_NOISEOEFF = (0.0, 1.0) 	       # Total throughput of the telescope+instrument, from photons arriving to the main mirror to photons being quantified by the detector [0:none to 1:perfect]. The user can provide wavelength dependent values as neff@wavelength[um] (e.g., '0.087@2.28,0.089@2.30,0.092@2.31,0.094@2.32,...')
RANGE_GENERATOR_NOISEOEMIS = (0.0, 1.0)        # Emissivity of the telescope+instrument optics [0 to 1]
RANGE_GENERATOR_NOISEOTEMP = (0.0, 1e4)        # Temperature of the telescope+instrument optics [K]
RANGE_GENERATOR_GCM_BINNING = (1, 1e4)         # Spatial binning applied to the GCM data when computing spectra. 1: Full resolution

EARTH_ORBITAL_PERIOD = 365.256363004           # [days]


LIST_STAR_TYPE = ('O', 'B', 'A', 'F', 'G', 'K', 'M')
RANGE_STAR_TEMPERATURE_BY_STAR_TYPE = ((3e4, 1e5), (1e4, 3e4), (7500, 1e4), (6000, 7500),
                                       (5200, 6000), (3700, 5200), (2400, 3700))
RANGE_STAR_RADIUS_BY_STAR_TYPE = ((6.6, 3000), (1.8, 6.6), (1.4, 1.8), (1.15, 1.4),
                                  (0.96, 1.15), (0.7, 0.96), (1e-3, 0.7))
RANGE_STAR_MASS_BY_STAR_TYPE = ((16, 250), (2.1, 16), (1.4, 2.1),
                                (1.04, 1.4), (0.8, 1.04), (0.45, 0.8), (0.08, 0.45))

DEFAULT_USER_DEFINED_TELESCOPE_PARAM = {
        'beam': 1,
        'beam_unit': 'diffrac',
        'gas_model': 'Y',
        'cont_model': 'Y',
        'range1': 0.4,
        'range2': 20,
        'rangeunit': 'um',
        'resolution': 200,
        'resolutionunit': 'RP',
        'radunits': 'rel',
        'diamtele': 6.5,
        'lograd': 'N',
        'cont_stellar': 'N',
        'trans_apply': 'N',
        'trans_show': 'N',
        'trans': '02-01',
        'noise': 'CCD',
        'noise2': 0.05,
        'noisetime': 0.2,
        'noiseotemp': 50,
        'noiseoeff': 0.2,
        'noiseoemis': 0.1,
        'noiseframes': 4000,
        'noisepixels': 8,
        'noise1': 16.8,
        'telescope': 'SINGLE',
        'telescope1': 1,
        'telescope2': 2.0,
        'telescope3': 1.0
    }


JWST_MIRI_LRS_TELESCOPE_PARAM = {
        'beam': 1,
        'beam_unit': 'diffrac',
        'gas_model': 'Y',
        'cont_model': 'Y',
        'range1': 5,
        'range2': 12,
        'rangeunit': 'um',
        'resolution': 100,
        'resolutionunit': 'RP',
        'radunits': 'pt',
        'diamtele': 5.64,
        'lograd': 'Y',
        'cont_stellar': 'Y',
        'trans_apply': 'N',
        'trans_show': 'N',
        'trans': '02-01',
        'noise': 'CCD',
        'noise2': 0.005,
        'noisetime': 1000,
        'noiseotemp': 50,
        'noiseoeff': 0.3,
        'noiseoemis': 0.1,
        'noiseframes': 10,
        'noisepixels': 8,
        'noise1': 32.6,
        'telescope': 'SINGLE',
        'telescope1': 1,
        'telescope2': 2.0,
        'telescope3': 1.0
    }

JWST_MIRI_MRS_TELESCOPE_PARAM = {
        'beam': 1,
        'beam_unit': 'diffrac',
        'gas_model': 'Y',
        'cont_model': 'Y',
        'range1': 5,
        'range2': 28.3,
        'rangeunit': 'um',
        'resolution': 2400,
        'resolutionunit': 'RP',
        'radunits': 'pt',
        'diamtele': 5.64,
        'lograd': 'Y',
        'cont_stellar': 'Y',
        'trans_apply': 'N',
        'trans_show': 'N',
        'trans': '02-01',
        'noise': 'CCD',
        'noise2': 0.005,
        'noisetime': 1000,
        'noiseotemp': 50,
        'noiseoeff': 0.3,
        'noiseoemis': 0.1,
        'noiseframes': 10,
        'noisepixels': 8,
        'noise1': 32.6,
        'telescope': 'SINGLE',
        'telescope1': 1,
        'telescope2': 2.0,
        'telescope3': 1.0
    }

JWST_NIRSPEC_PRISM_TELESCOPE_PARAM = {
        'range1': 0.70,
        'range2': 5.00,
        'rangeunit': 'um',
        'resolution': 100,
        'resolutionunit': 'RP',
        'telescope': 'SINGLE',
        'diamtele': 5.64,
        'beam': 1.0,
        'beam_unit': 'diffrac',
        'telescope1': 1,
        'telescope2': 2.0,
        'telescope3': 1.0,
        'noise': 'CCD',
        'noise1': 16.8,
        'noise2': 0.005,
        'noiseotemp': 50,
        'noiseoeff': 0.4,
        'noiseoemis': 0.10,
        'noisetime': 1000,
        'noiseframes': 10,
        'noisepixels': 8,
        'trans_apply': 'N',
        'trans_show': 'N',
        'lograd': 'Y',
        'gas_model': 'Y',
        'cont_model': 'Y',
        'cont_stellar': 'Y',
        'radunits': 'pt',
        'resolutionkernel': 'N',
        'trans': '02_01'}

JWST_NIRSPEC_1000_TELESCOPE_PARAM = {
        'range1': 1.00,
        'range2': 5.30,
        'rangeunit': 'um',
        'resolution': 1000,
        'resolutionunit': 'RP',
        'telescope': 'SINGLE',
        'diamtele': 5.64,
        'beam': 1.0,
        'beam_unit': 'diffrac',
        'telescope1': 1,
        'telescope2': 2.0,
        'telescope3': 1.0,
        'noise': 'CCD',
        'noise1': 16.8,
        'noise2': 0.005,
        'noiseotemp': 50,
        'noiseoeff': 0.3,
        'noiseoemis': 0.10,
        'noisetime': 1000,
        'noiseframes': 10,
        'noisepixels': 8,
        'trans_apply': 'N',
        'trans_show': 'N',
        'lograd': 'Y',
        'gas_model': 'Y',
        'cont_model': 'Y',
        'cont_stellar': 'Y',
        'radunits': 'pt',
        'resolutionkernel': 'N',
        'trans': '02-01'}


AVAILABLE_MOLECULES = {'He': 'HIT[0]', 'H2O': 'HIT[1]', 'CO2': 'HIT[2]', 'O3': 'HIT[3]', 'N2O': 'HIT[4]',
                       'CO': 'HIT[5]', 'CH4': 'HIT[6]', 'O2': 'HIT[7]', 'NO': 'HIT[8]', 'SO2': 'HIT[9]',
                       'NH3': 'HIT[11]', 'HNO3': 'HIT[12]', 'OH': 'HIT[13]', 'HF': 'HIT[14]', 'HCl': 'HIT[15]',
                       'HBr': 'HIT[16]', 'HI': 'HIT[17]', 'ClO': 'HIT[18]', 'OCS': 'HIT[19]', 'H2CO': 'HIT[20]',
                       'HOCl': 'HIT[21]', 'N2': 'HIT[22]', 'HCN': 'HIT[23]', 'CH3Cl': 'HIT[24]', 'H2O2': 'HIT[25]',
                       'C2H2': 'HIT[26]', 'PH3': 'HIT[28]', 'COF2': 'HIT[29]', 'SF6': 'HIT[30]', 'H2S': 'HIT[31]',
                       'HCOOH': 'HIT[32]', 'HO2': 'HIT[33]', 'O': 'HIT[34]', 'ClONO2': 'HIT[35]', 'NO+': 'HIT[36]',
                       'HOBr': 'HIT[37]', 'C2H4': 'HIT[38]', 'CH3OH': 'HIT[39]', 'CH3Br': 'HIT[40]',
                       'CH3CN': 'HIT[41]', 'CF4': 'HIT[42]', 'C4H2': 'HIT[43]', 'HC3N': 'HIT[44]', 'H2': 'HIT[45]',
                       'CS': 'HIT[46]', 'SO3': 'HIT[47]', 'COCl2': 'HIT[49]'}


CHOSEN_MOLECULES = {'H2O': 'HIT[1]', 'CH4': 'HIT[6]', 'NH3': 'HIT[11]', 'PH3': 'HIT[28]', 'N2O': 'HIT[4]',
                    'H2': 'HIT[45]', 'CO2': 'HIT[2]', 'H2S': 'HIT[31]', 'C2H6': 'HIT[27]',
                    'O3': 'HIT[3]', 'O2': 'HIT[7]', 'N2': 'HIT[22]'}

CHOSEN_MOLECULES_WEIGHTS = [18.02, 16.043, 17.02, 34.0, 44.013, 2.015, 44.01, 34.08, 30.0674, 47.998, 31.9988, 28.0134]
