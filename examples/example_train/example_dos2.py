import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


torch.set_default_dtype(torch.float64)

# Define Calculation for homonuclear silicon
#---------------------------------------------------
parameter_db_path = './data _tbmaltpaper/siband.hdf5'

shell_dict = {14: [0, 1, 2]}
species = [14]

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
class SiliconDataset(Dataset):
    def __init__(self, numbers, positions, latvecs, homo_lumos, eigenvalues):
        self.numbers = numbers
        self.positions = positions
        self.latvecs = latvecs
        self.homo_lumos = homo_lumos
        self.eigenvalues = eigenvalues

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, index):
        number = self.numbers[index] #atomic number
        position = self.positions[index]
        latvec = self.latvecs[index]
        homo_lumo = self.homo_lumos[index]
        eigenvalue = self.eigenvalues[index]
        system = {"number": number, 
                  "position": position, 
                  "latvec": latvec, 
                  "homo_lumo": homo_lumo, 
                  "eigenvalue": eigenvalue, 
                  "index": index
                  }
        return system

def create_dataset(path):
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        systems_idxs = len(f[key].keys())/6

        regex_pattern = r'(Si|C)'
        data = {'number': [[14 if atom == "Si" else 6 for atom in re.findall(regex_pattern,key)]],
                'position': torch.from_numpy(f[key]['1' + 'position'][:]).unsqueeze(dim=0),
                'lattice vector': torch.from_numpy(f[key]['1' + 'lattice vector'][:]).unsqueeze(dim=0),
                'homo_lumo': torch.from_numpy(f[key]['1' + 'homo_lumo'][:]).unsqueeze(dim=0),
                'eigenvalue': torch.from_numpy(f[key]['1' + 'eigenvalue'][:]).unsqueeze(dim=0),
                }
        for idx in range(2, int(systems_idxs) + 1):#index:
            data['number'].append(data['number'][0])
            for group in ['position', 'lattice vector', 'homo_lumo', 'eigenvalue']:
                data[group] = torch.cat((data[group], torch.from_numpy(f[key][str(idx) + group][:]).unsqueeze(dim=0)), dim=0)

        return SiliconDataset(torch.IntTensor(data['number']),
                              data['position'], 
                              data['lattice vector'], 
                              data['homo_lumo'],
                              data['eigenvalue']
                              )

dataset_Si63v_relax_pbe = create_dataset('./data_wenbo/dataset/fhi-aims_si63v_relax_pbe.hdf')
#dataset_Si63v_hse_101 = create_dataset('./data_wenbo/dataset/fhi-aims_si63v_hse_101.hdf')

# Energy window for dos sampling
#points = torch.linspace(-4.6, 6.9, 1151)
points = torch.linspace(-3.0, 2.0, 501)

#prepare training data
training_size = 1
indice = torch.arange(training_size).tolist()

data_train = dataset_Si63v_relax_pbe#[0]#[: training_size]
#data_train = dataset_Si63v_hse_101#[: training_size]
dataloader_train = DataLoader(data_train, batch_size=2)
print('DATALOADER')
#for batch, x in enumerate(dataloader_train):
#    print(x)
print('@@@@@@@@@@2')
#print(data_train['numbers'])
print(data_train.numbers)

#ref_ev, ref_hl = (data_train['eigenvalue'], data_train['homo_lumo'])
ref_ev, ref_hl = (data_train.eigenvalues, data_train.homo_lumos)
#reference data
targets = {'eigenvalues': ref_ev,
           'homo_lumos': ref_hl
           }

# Create Plot of training DOS reference
energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
#energies_plot = torch.linspace(-3, 2, 500).repeat(training_size, 1)
dos_ref_plot = dos((targets['eigenvalues']), energies_plot, 0.09)
dos_ref_plot_mean = dos_ref_plot.mean(dim=0)
dos_ref_plot_std = dos_ref_plot.std(dim=0)
fermi_train_plot = targets['homo_lumos'].mean(dim=-1)
energies_train_plot = fermi_train_plot.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(training_size, 0)

plt.plot(energies_plot[0], dos_ref_plot_mean, '-', linewidth=1.0)
plt.fill_between(energies_plot[0], dos_ref_plot_mean + dos_ref_plot_std, dos_ref_plot_mean - dos_ref_plot_std, alpha=0.5, facecolor='darkred')
plt.fill_between(energies_train_plot[0], -3, 80, alpha=0.2)

plt.tick_params(direction='in', labelsize='13', width=1.1, top='on', right='on', zorder=10)
plt.xlim((-18.2, 5.2))
#plt.xlim((-3.0, 2.0))
plt.ylim((-1, 70))
#plt.ylim((-1, 60))
plt.xlabel("Energy [eV]", fontsize=14)
plt.ylabel("DOS", fontsize=14)

#plt.plot(energies_plot[0] - fermi_train_plot.mean(dim=0), dos_plot_mean, linewidth=1.5)
#plt.fill_between(energies_train_plot.mean(dim=0) - fermi_train_plot.mean(dim=0), -3, 60, alpha=0.2)
plt.show()

# Construct geometry
#geometry = Geometry(data_train['number'], 
#                    data_train['position'],
#                    lattice_vector= data_train['latvec'],
#                    units='a',
#                    cutoff=torch.tensor([18.0])/length_units['angstrom']
#                    )
#print(geometry.atomic_numbers)
#orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)
#print(orbs)

# Define Training
#-----------------------------------------------------------

# Hellinger Loss function def
loss_func = hellinger_loss

#delegates
def prediction_data_delegate(calculator, targets, **kwargs):
    predictions = dict()
    fermi_dftb = calculator.homo_lumo.mean(dim=-1) / energy_units['ev']
    n_batch = 1
    energies_dftb = fermi_dftb.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(n_batch, 0)
    dos_dftb = dos(calculator.eig_values / energy_units['ev'], energies_dftb, 0.09)

    predictions['dos'] = dos_dftb
    return predictions

def reference_data_delegate(calculator, targets, **kwargs):
    reference = dict()
    ref_ev = targets['eigenvalues']
    fermi_train = targets['homo_lumos'].mean(dim=-1)
    energies_train = fermi_train.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(training_size, 0)
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

number_of_epochs = 30
#for epoch in range(number_of_epochs):
#    print(f"Epoch {epoch+1}/{number_of_epochs}")
#    _loss = 0
#    dftb_calculator(geometry, orbs, grad_mode='direct')
#    total_loss, _ = loss_entity(dftb_calculator, targets)
#    _loss += total_loss
#    optimizer.zero_grad()
#    _loss.retain_grad()
#    _loss.backward(retain_graph=True)
#    optimizer.step()
#    print(f"Loss: {_loss.item()}")

for epoch in range(number_of_epochs):
    print(f"Epoch {epoch+1}/{number_of_epochs}")
    train_loop(dataloader_train, optimizer, dftb_calculator)


#Plotting of result
#---------------------------------------------------

#Reference
ref_hl_plot = targets['homo_lumos']
ref_ev_plot = targets['eigenvalues']
ref_fermi_plot = targets['homo_lumos'].mean(dim=-1)
ref_energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
ref_dos_plot = dos((ref_ev_plot), ref_energies_plot, 0.09)

#Original DFTB calc
h_feed_o = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline)
s_feed_o = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=CubicSpline)
# Calculator
mix_params = {'mix_param': 0.2, 
              'init_mix_param': 0.2,
              'generations': 3,
              'tolerance': 1e-10
              }
kwargs = {}
kwargs['mix_params'] = mix_params
dftb_calculator_o = Dftb2(h_feed_o, s_feed_o, o_feed, u_feed, suppress_scc_error=True, filling_scheme=None, filling_temp=None, **kwargs)
dftb_calculator_o(geometry, orbs)

hl_dftb = getattr(dftb_calculator_o, 'homo_lumo').detach() / energy_units['ev']
fermi_dftb = hl_dftb.mean(-1)
eigval_dftb = dftb_calculator_o.eig_values.detach() / energy_units['ev']
dos_dftb = dos((eigval_dftb), energies_plot, 0.09)

# Results before training
plt.plot((ref_energies_plot - ref_fermi_plot).squeeze(0), ref_dos_plot.squeeze(0), label='DFT')
plt.plot((ref_energies_plot - fermi_dftb).squeeze(0), dos_dftb.squeeze(0), label='siband-1-1')
plt.fill_between(energies_train_plot[0], -3, 80, alpha=0.2)

plt.tick_params(direction='in', labelsize='13', width=1.1, top='on', right='on')

plt.xlim((-18.2, 5.2))
plt.ylim((-1, 70))

plt.rcParams["font.family"] = "arial"
plt.xlabel(r'E - $\mathregular{E_f}$ [eV]', fontsize=15)
plt.ylabel('DOS [states / eV]', fontsize=15)
plt.title("Before training", fontsize=13)
plt.legend(fontsize=13)
plt.show()

# Prediction after training
hl_pred = getattr(dftb_calculator, 'homo_lumo').detach() / energy_units['ev']
fermi_pred = hl_pred.mean(-1)
eigval_pred = dftb_calculator.eig_values.detach() / energy_units['ev']
dos_pred = dos((eigval_pred), ref_energies_plot, 0.09)
plt.plot((ref_energies_plot - ref_fermi_plot).squeeze(0), ref_dos_plot.squeeze(0), label='DFT')
plt.plot((ref_energies_plot - fermi_pred).squeeze(0), dos_pred.squeeze(0), label='spline')
#plt.xlim(-3.5, 2)
#plt.ylim(-2, 40)
plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                right='on')
plt.rcParams["font.family"] = "arial"
plt.xlabel(r'E - $\mathregular{E_f}$ [eV]', fontsize=15)
plt.ylabel('DOS [states / eV]', fontsize=15)
plt.title("After training", fontsize=13)
plt.legend(fontsize=13)
plt.show()





