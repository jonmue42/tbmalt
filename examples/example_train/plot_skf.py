import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

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


# Load (unpickle) the instance from the file
with open("h_feed_o.pkl", "rb") as f:
    h_feed = pickle.load(f)

plot_interpolation(h_feed)



