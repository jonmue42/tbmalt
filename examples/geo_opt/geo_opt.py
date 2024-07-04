from os.path import exists
from typing import Any
import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline

from ase.build import molecule

Tensor = torch.Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

molecule_names = 'H2'

shell_dict = {1: [0], 8: [0, 1]}

parameter_db_path = 'example_dftb_parameters.h5'

#geometry = Geometry.from_ase_atoms(molecule(molecule_names))
#print(geometry.atomic_numbers)

coords = torch.tensor([[0.00, 0.00, 0.00],
                [0.00, 0.00, 0.95]], requires_grad=True)

geometry = Geometry(torch.tensor([1, 1], dtype=torch.int64),
                    coords.clone(),
                    )


#geometry = Geometry(torch.tensor([1, 1], dtype=torch.int64),
#                     torch.tensor([[0.00, 0.00, 0.00],
#                            [0.00, 0.00, 0.95]], requires_grad=True))

orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)

species = torch.unique(geometry.atomic_numbers)
species = species[species != 0].tolist()

h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', interpolation='spline')
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', interpolation='spline')

o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

u_feed = HubbardFeed.from_database(parameter_db_path, species)

r_feed = RepulsiveSplineFeed.from_database(parameter_db_path, species)


dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed)#, r_feed)
#dftb_calculator(geometry, orbs)

#
#def loss_fn(calculator):
#    return dftb_calculator.mermin_energy
#
lr = 0.002
#params = [{'params': geometry._positions, 'lr': lr}] <- This doesnt work as geometry._positions is not a non-leaf tensor
params = [{'params': coords, 'lr': lr}]  #This doesnt work as there is a inplace operation happens somewhere
optimizer = torch.optim.Adam(params, lr=lr)

epochs = 100
for epoch in range(epochs):

    loss = dftb_calculator(geometry, orbs)
    #loss = loss_fn(coords)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')



