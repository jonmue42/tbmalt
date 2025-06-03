import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline, test_iter, test_iter2
from tbmalt.physics.dftb import Dftb2
import tbmalt.common.maths as tbmalt_math
from tbmalt.ml.loss_function import Loss, hellinger_loss
from tbmalt import Geometry, OrbitalInfo
from tbmalt.data.units import energy_units, length_units

from tbmalt.physics.dftb.properties import dos

from plot_dos import plot_dos, plot_training_ref, plot_dos_test, plot_interpolation
from Si_geo import geo_si


torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)

#parameter_db_path = "./data/auorg.h5"
parameter_db_path = "./data/siband.hdf5"

#shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
shell_dict = {14: [0, 1, 2]}

#species = [8, 1]
species = [14]


h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=test_iter)
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=test_iter)
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)
u_feed = HubbardFeed.from_database(parameter_db_path, species)

#for key, interpolator_o in s_feed._off_sites.items():
#    plot_interpolation(interpolator_o, 'Before training')

for key, interpolator in s_feed._off_sites.items():
    print('key: ', key)
    if key == key: #'(14, 14, 2, 2)':
        plot_interpolation(interpolator, 'Before training: ' + key)
        #print(interpolator.coefficients * 0.1)
        print('Coeffs:', interpolator._coefficients)
        #randoms_nums = torch.randn_like(interpolator._coefficients)
        randoms_nums = torch.rand_like(interpolator._coefficients)
        #print('randoms_nums: ', randoms_nums)
        interpolator._coefficients = interpolator._coefficients +  randoms_nums*5e-6#+ interpolator._coefficients * 0.5 
        #interpolator._coefficients = torch.tensor([[ 2.9388112418e-11,  3.2655209604e+01, -6.6276478700e+00, 5.9749735405e-01, -2.0591706295e-02],
        #                                           [-1.6969009248e-03,  3.3416785422e+00, -6.5507170489e-01, 4.8476451819e-02, -1.6718022028e-03],
        #                                           [ 1.1189186100e+00, -1.0603312293e-01, -6.1086122159e-02,-3.7361090985e-03,  1.3049943146e-04]])
        #print(interpolator.coefficients)
        plot_interpolation(interpolator, 'After training: ' + key)

#for key, interpolator_o in s_feed._off_sites.items():
#    plot_interpolation(interpolator_o, 'After training')
#
# Calculator
mix_params = {'mix_param': 0.2, 
              'init_mix_param': 0.2,
              'generations': 3,
              'tolerance': 1e-10
              }
kwargs = {}
kwargs['mix_params'] = mix_params
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, suppress_scc_error=True, filling_scheme=None, filling_temp=None, **kwargs)

#geometry = Geometry(
#        torch.tensor([8, 1, 1]), 
#        torch.tensor([[0.0, 0.0, 0.0],
#                      [0.0, 1.0, 1.0],
#                     [0.0, 1.0, -12.0]]),
#               #units='a'
#               units='bohr'
#               )
geometry = Geometry(
        geo_si['atomic_nums'], 
        geo_si['position'],
        lattice_vector=geo_si['latvec'],
        units='a',
        cutoff=torch.tensor([18.0])/length_units['angstrom']
               )


print('Distances: ', geometry.distances)
print('Maximum distance: ', geometry.distances.max())
#Get minimum distance bigger than 0.0
min_dist = geometry.distances[geometry.distances > 0.0].min()
print('Minimum distance: ', min_dist)

orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)

energy = dftb_calculator(geometry, orbs, grad_mode='direct')
energy = energy 
print("Energy: ", energy.item())

