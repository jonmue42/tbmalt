import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

# File with the sk data
path = './auorg.hdf5'

species = [1, 8]

shell_dict = {1: [0], 8: [0, 1]}

# Set up geometry
H2O = Geometry(torch.tensor([1, 8, 8]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 1.0, -0.5],
                             [0.0, -1.0, 0.5]], requires_grad=False)
               )

orbital_info = OrbitalInfo(H2O.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

# Set up the calculator
dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)

# Run a SCF calculation
dftb_calculator(H2O, orbital_info)

# Get total energy at position
total_energy = dftb_calculator.total_energy

# Set up geometry shifted by a delta
delta = 1e-3
H2O_dx1 = Geometry(torch.tensor([1, 8, 8]), 
               torch.tensor([[0.0 + delta, 0.0, 0.0],
                             [0.0, 1.0, -0.5],
                             [0.0, -1.0, 0.5]], requires_grad=False)
               )

orbital_info_dx1 = OrbitalInfo(H2O.atomic_numbers, shell_dict, shell_resolved=False)

# Run calculation again
dftb_calculator(H2O_dx1, orbital_info_dx1)

# Get total energy at position
total_energy_dx1 = dftb_calculator.total_energy


# Calculate derivative via finite difference
grad = (total_energy_dx1 - total_energy) / delta

print('Finite difference gradient:', grad)
