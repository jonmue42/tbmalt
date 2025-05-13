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

from training_vars import training_globals, dataset_vars
from silicon_dataset import SiliconDataset
from plot_dos import plot_dos, plot_training_ref, plot_dos_test, plot_interpolation
from scipy.interpolate import CubicSpline as ScipyCubicSpline

torch.set_default_dtype(torch.float64)

#create a few example points for spline
x = torch.linspace(0, 5, 6)
x_np = np.linspace(0, 5, 6)
print(x)
print(x_np)
y = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), requires_grad=False)
y_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

scipy_interpolator = ScipyCubicSpline(x_np, y)
print(scipy_interpolator.c)
plt.plot(x_np, y_np, 'o', label='data points')
x_scipy = np.linspace(0, 5, 1000)
y_scipy = scipy_interpolator(x_scipy)
plt.plot(x_scipy, y_scipy, label='scipy interpolator')
plt.legend()
plt.show()

for idx in range(len(y)):
    if idx % 2 == 0:
        scipy_interpolator.c = np.array([[4., 0., 3., 1., 0.],
 [0., 2., 0., 0., 0.],
 [1., 1., 1., 1., 6.],
 [1., 2., 3., 4., 5.]])

print(scipy_interpolator.c)
plt.plot(x_np, y_np, 'o', label='data points')
x_scipy = np.linspace(0, 5, 1000)
y_scipy = scipy_interpolator(x_scipy)
plt.plot(x_scipy, y_scipy, label='scipy interpolator')
plt.legend()
plt.show()



interpolator = CubicSpline(x, y, tail=0.0)

##load pickle of interpolator
#with open('interpolators/(14, 14, 0, 0)interpolator_hfeed.pkl', 'rb') as f:
#    interpolator = pickle.load(f)
#coeffs = torch.load('coeffs/(14, 14, 0, 0)coeffs_hfeed.pt')
#
plot_interpolation(interpolator, "Before")

for idx in range(len(y)):
    if idx % 2 == 0:
        interpolator.y[idx] += 2.0


plot_interpolation(interpolator, "Before")
#
#interpolator._coefficients = coeffs.detach()
#plot_interpolation(interpolator, "Before")
