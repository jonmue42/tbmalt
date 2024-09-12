"""
Geometry optimization of H2O using pytorch
"""
import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#File with the sk data
database_path = './auorg.hdf5'
path = database_path


#######
#ATOM SETUP
#######
# species occuring in H2O
species = torch.tensor([1, 8])
# relevant shells
shell_dict = {1: [0], 8: [0, 1]}

# set up the starting geometry
H2O = Geometry(torch.tensor([1, 8, 8]),
               torch.tensor([[0.0, 0.0, 0.0], 
                             [0.0, 1.0, -0.5], 
                             [0.0, -1.0, -0.5]], requires_grad=False))
H2O._positions.requires_grad = True

orbital_info = OrbitalInfo(H2O.atomic_numbers, shell_dict, shell_resolved=False)


######
# set up the feed objects
######
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')

overlap_feed = SkFeed.from_database(path, species, 'overlap')

occupation_feed = SkfOccupationFeed.from_database(path, species)

hubbard_feed = HubbardFeed.from_database(path, species)

repulsion_feed = RepulsiveSplineFeed.from_database(path, species)

######
# set up the calculator
######
#dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsion_feed)
dftb_calculator(H2O, orbital_info)


######
# set hyperparameters
######
learning_rate = 0.001
epochs = 100

optimizer = torch.optim.Adam([H2O._positions], lr=learning_rate)

######
# training loop
######
#for epoch in range(epochs):
#    print(epoch)
#    # forward pass
#    dftb_calculator(H2O, orbital_info)
#    loss = loss_fn(dftb_calculator)
#
#    # backward pass
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    if epoch % 100 == 0:
#        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
#
#


