import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline, ExponentialInter, test_iter
from tbmalt.physics.dftb import Dftb2
import tbmalt.common.maths as tbmalt_math
from tbmalt.ml.loss_function import Loss, hellinger_loss
from tbmalt import Geometry, OrbitalInfo
from tbmalt.data.units import energy_units, length_units

from tbmalt.physics.dftb.properties import dos

from training_vars import training_globals, dataset_vars
from silicon_dataset import SiliconDataset
from plot_dos import plot_dos, plot_training_ref, plot_dos_test, plot_interpolation


torch.set_default_dtype(torch.float64)

# Define Calculation for homonuclear silicon
#---------------------------------------------------
dataset_name = 'si63v_hse_101'
#dataset_name = 'si32c31_hse_82'
#dataset_vars = dataset_vars[dataset_name]

#parameter_db_path = './data _tbmaltpaper/siband.hdf5'
parameter_db_path = dataset_vars[dataset_name]['parameter_db_path']

shell_dict = {14: [0, 1, 2]}
#shell_dict = {14: [0, 1], 6: [0, 1]}
#species = [14, 6] # Si, C
species = [14] # Si, C

# Feeds
#h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=test_iter)#, requires_grad_offsite=True), requires_grad_onsite=True,)
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline)#, requires_grad_offsite=True), requires_grad_onsite=True,)

#s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=test_iter)#, requires_grad_offsite=True, requires_grad_onsite=True,)
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=CubicSpline)#, requires_grad_offsite=True, requires_grad_onsite=True,)

o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

u_feed = HubbardFeed.from_database(parameter_db_path, species)

# Calculator
mix_params = {'mix_param': 0.2, 
              'init_mix_param': 0.2,
              'generations': 3,
              'tolerance': 1e-10
              }
kwargs = {}
kwargs['mix_params'] = mix_params
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, suppress_scc_error=True, filling_scheme=None, filling_temp=None, **kwargs)

# Prepare Data
#---------------------------------------------------
dataset = SiliconDataset.create_dataset(dataset_vars[dataset_name]['dataset_path'])

# Energy window for dos sampling
points = dataset_vars[dataset_name]['points']

#prepare training data
training_size, test_size = dataset_vars[dataset_name]['training_split'][0], dataset_vars[dataset_name]['training_split'][1]
indice = torch.arange(training_size).tolist()

data_subset_train, data_subset_test, _ = random_split(dataset, dataset_vars[dataset_name]['training_split'])
train_indeces = data_subset_train.indices
data_train = dataset[train_indeces]

batch_size_train = dataset_vars[dataset_name]['batch_size_train']
batch_size_test = dataset_vars[dataset_name]['batch_size_test']
dataloader_train = DataLoader(data_subset_train, batch_size=batch_size_train)
dataloader_test = DataLoader(data_subset_test, batch_size=batch_size_test)

ref_ev, ref_hl = (data_train['eigenvalue'], data_train['homo_lumo'])
#reference data
targets = {'eigenvalues': ref_ev,
           'homo_lumos': ref_hl
           }

# Create Plot of training DOS reference
plot_training_ref(targets, training_size, points)

# Define Training
#-----------------------------------------------------------

# Hellinger Loss function def
loss_func = hellinger_loss

#delegates
def prediction_data_delegate(calculator, targets, batch_size, **kwargs):
    predictions = dict()
    fermi_dftb = calculator.homo_lumo.mean(dim=-1) / energy_units['ev']
    energies_dftb = fermi_dftb.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(batch_size, 0)
    dos_dftb = dos(calculator.eig_values / energy_units['ev'], energies_dftb, training_globals['dos_sigma'])

    predictions['dos'] = dos_dftb
    return predictions

def reference_data_delegate(calculator, targets, batch_size, **kwargs):
    reference = dict()
    ref_ev = targets['eigenvalues']
    fermi_train = targets['homo_lumos'].mean(dim=-1)
    energies_train = fermi_train.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(batch_size, 0)
    dos_ref = dos((ref_ev), energies_train, training_globals['dos_sigma'])

    reference['dos'] = dos_ref
    return reference

# Define Loss entity
loss_entity = Loss(prediction_data_delegate, reference_data_delegate, loss_functions=loss_func, reduction='sum')

# Define params to optimize (in this case H and S offsites)
for key in h_feed._off_sites.keys():
    h_feed._off_sites[key].y.requires_grad_(True)
    s_feed._off_sites[key].y.requires_grad_(True)

h_var = [val.y for key, val in h_feed._off_sites.items()] #man kann auch nur ueber values laufen
s_var = [val.y for key, val in s_feed._off_sites.items()]
print('Svar')
print(s_var)
print(len(s_var))
print(s_var[0].size())
print(s_var[1].size())
print(s_var[2].size())
print(s_var[3].size())
print(s_var[4].size())
print(s_var[5].size())
print('Hvar')
print(h_var)
print(len(h_var))
print(h_var[0].size())
print(h_var[1].size())
print(h_var[2].size())
print(h_var[3].size())
print(h_var[4].size())
print(h_var[5].size())
print('Hfeed offsite')
print(h_feed._off_sites)
params = h_var + s_var

# optimizer
learning_rate = training_globals['learning_rate']
optimizer = torch.optim.Adam(params=params, lr=learning_rate)

# Training
#--------------------------------------------------
loss_list = []
def train_loop(dataloader, optimizer, dftb_calculator):
    _loss = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        targets = {'eigenvalues': data['eigenvalue'],
                   'homo_lumos': data['homo_lumo']
                   }

        geometry = Geometry(data['number'], 
                    data['position'],
                    lattice_vector= data['latvec'],
                    units='a',
                    cutoff=dataset_vars[dataset_name]['cutoff']
                    )
        orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)

        dftb_calculator(geometry, orbs, grad_mode='direct')

        loss, _ = loss_entity(dftb_calculator, targets, batch_size=batch_size_train)
        _loss = _loss + loss
    optimizer.zero_grad()
    _loss.retain_grad()
    _loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Training Loss: {_loss.item()}")
    loss_list.append(_loss.detach())

test_loss_list = []
def test_loop(dataloader, dftb_calculator):
    _loss = 0
    for batch, data in enumerate(dataloader):
        targets = {'eigenvalues': data['eigenvalue'],
                   'homo_lumos': data['homo_lumo']
                   }

        geometry_test = Geometry(data['number'], 
                                 data['position'],
                                 lattice_vector=data['latvec'],
                                 units='a',
                                 cutoff=dataset_vars[dataset_name]['cutoff']
                                 )
        orbs_test = OrbitalInfo(geometry_test.atomic_numbers, shell_dict, shell_resolved=False)

        dftb_calculator(geometry_test, orbs_test, grad_mode='direct')
        
        loss, _ = loss_entity(dftb_calculator, targets, batch_size=batch_size_test)
        _loss = _loss + loss
    print(f"Test Lost: {_loss.item()}")
    test_loss_list.append(_loss.detach())
    
number_of_epochs = training_globals['number_of_epochs']
for epoch in range(number_of_epochs):
    print(f"Epoch {epoch+1}/{number_of_epochs}")
    train_loop(dataloader_train, optimizer, dftb_calculator)
#    with torch.no_grad():
#        test_loop(dataloader_test, dftb_calculator)


#Plotting of result
#---------------------------------------------------
with torch.no_grad():
    ##Reference
    
    ##Original DFTB calc
    #h_feed_o = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=test_iter)
    h_feed_o = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline)
    #s_feed_o = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=test_iter)
    s_feed_o = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=CubicSpline)
    
    geometry_o = Geometry(data_train['number'],
                          data_train['position'],
                          lattice_vector=data_train['latvec'],
                          units='a',
                          cutoff=dataset_vars[dataset_name]['cutoff']
                          )
    orbs_o = OrbitalInfo(geometry_o.atomic_numbers, shell_dict, shell_resolved=False)
    
    # Calculator
    mix_params = {'mix_param': 0.2, 
                  'init_mix_param': 0.2,
                  'generations': 3,
                  'tolerance': 1e-10
                  }
    kwargs = {}
    kwargs['mix_params'] = mix_params
    dftb_calculator_o = Dftb2(h_feed_o, s_feed_o, o_feed, u_feed, suppress_scc_error=True, filling_scheme=None, filling_temp=None, **kwargs)
    
    #plot_dos(targets, training_size, geometry_o, orbs_o, dftb_calculator_o, points, labels=('DFT', 'siband-1-1'), title='Before training')
    
    # Prediction after training
    
    #plot_dos(targets, training_size, geometry_o, orbs_o, dftb_calculator, points, labels=('DFT', 'spline'), title='After training')
    
    # Plot test set
    #plot_dos_test(dataloader_test, test_size, batch_size_test, dftb_calculator_o, shell_dict, points, labels=('DFT', 'siband-1-1'), title='Before training')
    
    #plot_dos_test(dataloader_test, test_size, batch_size_test, dftb_calculator, shell_dict, points, labels=('DFT', 'spline'), title='After training')
    for key, interpolator_o in h_feed_o._off_sites.items():
        plot_interpolation(interpolator_o, interpolator_o, training_size, 'Before training')

    for key, interpolator in h_feed._off_sites.items():
        plot_interpolation(interpolator, interpolator, training_size, 'After training')
