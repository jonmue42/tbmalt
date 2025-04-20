import torch
from torch.utils.data import DataLoader, Dataset, random_split

import h5py
import re

torch.set_default_dtype(torch.float64)
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

    @classmethod
    def create_dataset(cls, path):
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
    
            return cls(torch.IntTensor(data['number']),
                                  data['position'], 
                                  data['lattice vector'], 
                                  data['homo_lumo'],
                                  data['eigenvalue']
                                  )
    
    
