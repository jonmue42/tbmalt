import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.physics.dftb import Dftb2
import tbmalt.common.maths as tbmalt_math
from tbmalt.ml.loss_function import Loss, hellinger_loss
from tbmalt import Geometry, OrbitalInfo
from tbmalt.data.units import energy_units, length_units

from tbmalt.physics.dftb.properties import dos

#from tbmalt.io.loadhdf import LoadHdf

from silicon_dataset import SiliconDataset
from plot_dos import plot_dos, plot_training_ref


torch.set_default_dtype(torch.float64)

# Define Calculation for homonuclear silicon
#---------------------------------------------------
parameter_db_path = './data _tbmaltpaper/siband.hdf5'

shell_dict = {14: [0, 1, 2]}
#shell_dict = {14: [0, 1, 2], 6: [0, 1, 3]}
#species = [14, 6] # Si, C
species = [14] # Si, C

# Feeds
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline)#, requires_grad_offsite=True), requires_grad_onsite=True,)

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

#dataset_Si63v_relax_pbe = create_dataset('./data_wenbo/dataset/fhi-aims_si63v_relax_pbe.hdf')
#dataset_Si63v_hse_101 = create_dataset('./data_wenbo/dataset/fhi-aims_si63v_hse_101.hdf')
dataset_Si63v_hse_101 =  SiliconDataset.create_dataset('./data_wenbo/dataset/fhi-aims_si63v_hse_101.hdf')
#dataset_Si32c31_hse_82 = create_dataset('./data_wenbo/dataset/fhi-aims_si32c31_hse_82.hdf')
#dataset_Si65_interstitial_hse = create_dataset('./data_wenbo/dataset/fhi-aims_si65_interstitial_hse.hdf')

dataset = dataset_Si63v_hse_101

# Energy window for dos sampling
#points = torch.linspace(-4.6, 6.9, 1151)
points = torch.linspace(-3.0, 2.0, 501)

#prepare training data
training_size = 2
indice = torch.arange(training_size).tolist()

data_subset_train = random_split(dataset, [2, 99])[0]
#data_subset_train = random_split(dataset_Si63v_relax_pbe, [1, 0])[0]
#data_subset_train = random_split(dataset_Si65_interstitial_hse, [0.5, 0.5])[0]
train_indeces = data_subset_train.indices
data_train = dataset[train_indeces]
#data_train = dataset_Si63v_relax_pbe[train_indeces]
#data_train = dataset_Si65_interstitial_hse[train_indeces]
print('DATATRAIN')
print(data_train['number'])
n_batch = 1
dataloader_train = DataLoader(data_subset_train, batch_size=n_batch)
print('DATALOADER')
#for batch, x in enumerate(dataloader_train):
#    print(x)
print('@@@@@@@@@@2')
#print(data_train.numbers)
print(data_train['number'])

#ref_ev, ref_hl = (data_train['eigenvalue'], data_train['homo_lumo'])
#ref_ev, ref_hl = (data_train.eigenvalues, data_train.homo_lumos)
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
def prediction_data_delegate(calculator, targets, **kwargs):
    predictions = dict()
    fermi_dftb = calculator.homo_lumo.mean(dim=-1) / energy_units['ev']
    energies_dftb = fermi_dftb.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(n_batch, 0)
    dos_dftb = dos(calculator.eig_values / energy_units['ev'], energies_dftb, 0.09)

    predictions['dos'] = dos_dftb
    return predictions

def reference_data_delegate(calculator, targets, **kwargs):
    reference = dict()
    ref_ev = targets['eigenvalues']
    fermi_train = targets['homo_lumos'].mean(dim=-1)
    energies_train = fermi_train.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(n_batch, 0)
    dos_ref = dos((ref_ev), energies_train, 0.09)

    reference['dos'] = dos_ref
    return reference

# Define Loss entity
loss_entity = Loss(prediction_data_delegate, reference_data_delegate, loss_functions=loss_func, reduction='sum')

# Define params to optimize (in this case H and S offsites)
for key in h_feed._off_sites.keys():
    h_feed._off_sites[key].coefficients.requires_grad_(True)
    s_feed._off_sites[key].coefficients.requires_grad_(True)

h_var = [val.coefficients for key, val in h_feed._off_sites.items()] #man kann auch nur ueber values laufen
s_var = [val.coefficients for key, val in s_feed._off_sites.items()]
params = h_var + s_var

# optimizer
learning_rate = 0.00005
optimizer = torch.optim.Adam(params=params, lr=learning_rate)

# Training
#--------------------------------------------------
def train_loop(dataloader, optimizer, dftb_calculator):
    for batch, data in enumerate(dataloader):
        loss = 0
        optimizer.zero_grad()
        targets = {'eigenvalues': data['eigenvalue'],
                   'homo_lumos': data['homo_lumo']
                   }

        geometry = Geometry(data['number'], 
                    data['position'],
                    lattice_vector= data['latvec'],
                    units='a',
                    cutoff=torch.tensor([18.0])/length_units['angstrom']
                    )
        orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)

        dftb_calculator(geometry, orbs, grad_mode='direct')

        loss, _ = loss_entity(dftb_calculator, targets)
        loss.retain_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f"Loss: {loss.item()}")

number_of_epochs = 1
for epoch in range(number_of_epochs):
    print(f"Epoch {epoch+1}/{number_of_epochs}")
    train_loop(dataloader_train, optimizer, dftb_calculator)


#Plotting of result
#---------------------------------------------------

##Reference

##Original DFTB calc
h_feed_o = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline)
s_feed_o = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=CubicSpline)

geometry_o = Geometry(data_train['number'],
                      data_train['position'],
                      lattice_vector=data_train['latvec'],
                      units='a',
                      cutoff=torch.tensor([18.0])/length_units['angstrom']
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

plot_dos(targets, training_size, geometry_o, orbs_o, dftb_calculator_o, points, labels=('DFT', 'siband-1-1'), title='Before training')

# Prediction after training

plot_dos(targets, training_size, geometry_o, orbs_o, dftb_calculator, points, labels=('DFT', 'spline'), title='After training')




