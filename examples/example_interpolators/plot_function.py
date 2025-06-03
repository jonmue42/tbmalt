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

from tbmalt.common.batch import pack, bT
from tbmalt.physics.dftb.properties import dos

from plot_dos import plot_dos, plot_training_ref, plot_dos_test, plot_interpolation
from Si_geo import geo_si


def fit_func(x, aa, bb, cc, dd, ee):
    return aa*torch.exp(bb * x + cc * x**2 + dd * x**3 + ee * x**4)


torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)

#x_o = interpolator.xp.detach()
#y_o = interpolator.y.detach()
#plt.plot(x_o, y_o, 'o', label='original')
#print('x_o:', x_o.shape)
#x_test = torch.linspace(0.8, 18, 87)
#plt.plot(x_test, y_o[:87], 'o', label='original_test')

 
#x = torch.linspace(x_o[0], x_o[-1]+5, 10000)
x = torch.linspace(0, 11, 1000)
#x = torch.linspace(0.8, 18, 10000)
print('x:', x.shape)
#aa = torch.tensor([2.9388112418e-27])
aa = torch.tensor([2.9388112418e-11])
bb = torch.tensor([3.2655202604e+01])
cc = torch.tensor([-6.6276478700e+00])
dd = torch.tensor([5.9749735405e-01])
ee = torch.tensor([-2.0591706295e-02])
y_inter = bT(fit_func(x, aa, bb, cc, dd, ee))
plt.plot(x, y_inter, '.-', label='interpolated')
plt.xlim((0, 11))
#plt.title(title)
plt.legend()

#y_inter = interpolator.forward(x_o)
#plt.plot(x_o, y_inter, '-', label='interpolated')

plt.show()

print('result:')
x = 6.0
print(2.9388112418e-11 * np.exp( 3.2655202604e+01 * x -  6.6276478700e+00 * x**2 + 5.9749735405e-01 * x**3 - 2.0591706295e-02 * x**4 ) )



