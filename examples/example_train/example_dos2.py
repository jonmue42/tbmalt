import torch
from torch.utils.data import DataLoader, Dataset

import h5py
import numpy as np
import re

from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.physics.dftb import Dftb2

#from tbmalt.io.loadhdf import LoadHdf


torch.set_default_dtype(torch.float64)

# Define Calculation for homonuclear silicon
#---------------------------------------------------
parameter_db_path = './data _tbmaltpaper/siband.hdf5'

shell_dict = {14: [0, 1, 2]}
species = [14]

# Feeds
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline, requires_grad_offsite=True)

s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation=CubicSpline, requires_grad_offsite=True)

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
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, suppress_scc_error=True, **kwargs)

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

def load_data(path):
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        print(key)
        print(len(f[key].keys())/6)
        systems_idxs = len(f[key].keys())/6

        regex_pattern = r'(Si|C)'
        data = {'number': [[14 if atom == "Si" else 6 for atom in re.findall(regex_pattern,key)]],
                'position': torch.from_numpy(f[key]['1' + 'position'][:]).unsqueeze(dim=0),
                'lattice vector': torch.from_numpy(f[key]['1' + 'lattice vector'][:]).unsqueeze(dim=0),
                'homo_lumo': torch.from_numpy(f[key]['1' + 'homo_lumo'][:]).unsqueeze(dim=0),
                'eigenvalue': torch.from_numpy(f[key]['1' + 'eigenvalue'][:]).unsqueeze(dim=0),
                }
        print(data)
        for idx in range(2, int(systems_idxs) + 1):#index:
            data['number'].append(data['number'][0])
            for group in ['position', 'lattice vector', 'homo_lumo', 'eigenvalue']:
                data[group] = torch.cat((data[group], torch.from_numpy(f[key][str(idx) + group][:]).unsqueeze(dim=0)), dim=0)

        return SiliconDataset(data['number'], 
                              data['position'], 
                              data['lattice vector'], 
                              data['homo_lumo'],
                              data['eigenvalue']
                              )
            


            

load_data('./data_wenbo/dataset/fhi-aims_si63v_hse_101.hdf')
