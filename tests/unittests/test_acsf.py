#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acsf module unit-tests."""
import torch
import pytest
import pkg_resources
from dscribe.descriptors import ACSF
from ase.build import molecule
from tbmalt.ml.acsf import Acsf
from tbmalt import Geometry
from tbmalt.data.elements import chemical_symbols
from tbmalt.common.batch import pack

dscribe_version = pkg_resources.get_distribution('dscribe').version

if dscribe_version == "1.2.2":
    pytestmark = pytest.mark.skip(
        "Skipping tests: Deprecated dscribe package detected")

# Set some global parameters which only used here
torch.set_default_dtype(torch.float64)
ch4 = molecule('CH4')
nh3 = molecule('NH3')
h2o = molecule('H2O')
h2o2 = molecule('H2O2')
h2 = molecule('H2')
cho = molecule('CH3CHO')
text = 'tolerance check'
textd = 'Device persistence check'


def test_single_g1(device):
    """Test G1 values in single geometry."""
    rcut = 6.0

    # 1. Molecule test, test for element resolved
    geo = Geometry.from_ase_atoms(ch4)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut, element_resolve=True)
    acsf()

    # Get reference for unittest
    acsf_t = ACSF(species=species, rcut=rcut)
    acsf_t_g = torch.from_numpy(acsf_t.create(ch4))

    assert torch.max(abs(acsf_t_g - acsf.g)) < 1E-6, text
    assert acsf.g.device == device, textd

    acsf_sum = Acsf(geo, g1_params=rcut, element_resolve=False)
    acsf_sum()
    assert torch.max(abs(acsf_t_g.sum(-1) - acsf_sum.g)) < 1E-6, text
    assert acsf_sum.g.device == device, textd

    # 2. Periodic system test
    ch4.cell = [1, 3, 3]
    geo = Geometry.from_ase_atoms(ch4)
    species = geo.chemical_symbols
    acsfp = Acsf(geo, g1_params=rcut, element_resolve=True)
    acsfp()

    # Get reference
    acsf_tp = ACSF(species=species, rcut=rcut, periodic=True)
    acsf_t_tp = torch.from_numpy(acsf_tp.create(ch4))

    assert torch.max(abs(acsf_t_tp - acsfp.g)) < 1E-6, text
    assert acsfp.g.device == device, textd


def test_batch_g1(device):
    """Test G1 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o])
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, g1_params=rcut)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut)
    acsf_d_g1 = acsf_d.create([ch4, h2o])

    assert torch.max(abs(torch.from_numpy(acsf_d_g1[0]) -
                         acsf.g[: acsf_d_g1[0].shape[0]])) < 1E-6, text
    assert torch.max(abs(torch.from_numpy(acsf_d_g1[1]) -
                         acsf.g[acsf_d_g1[0].shape[0]:])) < 1E-6, text
    assert acsf.g.device == device, textd

    acsf_sum = Acsf(geo, g1_params=rcut, element_resolve=False)
    acsf_sum()
    assert torch.max(abs(torch.cat([
        torch.from_numpy(ii) for ii in acsf_d_g1]).sum(-1) - acsf_sum.g)) <\
           1E-6, text
    assert acsf_sum.g.device == device, textd


def test_single_g2(device):
    """Test G2 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut,
                g2_params=torch.tensor([0.5, 1.0]), element_resolve=True)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g2_params=[[0.5, 1.0]])
    acsf_d_g1 = torch.from_numpy(acsf_d.create(ch4))

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d_g1[:, [1, 3]] - acsf.g[:, 2:])) < 1E-6, text


def test_batch_g2(device):
    """Test G2 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o])
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, g1_params=rcut,
                g2_params=torch.tensor([0.5, 1.0]),
                element_resolve=True, atom_like=False)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g2_params=[[0.5, 1.0]])
    acsf_d_g1 = pack([torch.from_numpy(ii) for ii in acsf_d.create([ch4, h2o])])

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d_g1[..., :4] - acsf.g[..., [0, 3, 1, 4]])) < 1E-6, text


def test_single_g3(device):
    """Test G3 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut,
                g3_params=torch.tensor([1.0]),
                element_resolve=True)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g3_params=[1.0])
    acsf_d = torch.from_numpy(acsf_d.create(ch4))

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d[:, [1, 3]] - acsf.g[:, 2:])) < 1E-6, text


def test_batch_g3(device):
    """Test G3 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o])
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, g1_params=rcut,
                g3_params=torch.tensor([1.0]),
                element_resolve=True, atom_like=False)
    g = acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g3_params=[ 1.0])
    acsf_d = pack([torch.from_numpy(ii) for ii in acsf_d.create([ch4, h2o])])

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(
        acsf_d[..., :4] - g[..., [0, 3, 1, 4]])) < 1E-6, text


def test_single_g4(device):
    """Test G4 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()

    acsf_d = ACSF(species=species, rcut=rcut, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = torch.from_numpy(acsf_d.create(ch4))

    assert torch.max(abs(acsf_d_g4 - g)) < 1E-6, text


def test_cho_g4(device):
    """Test G4 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(cho)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True)
    g = acsf()

    acsf_d = ACSF(species=species, rcut=rcut, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = torch.from_numpy(acsf_d.create(cho))

    assert torch.max(abs(
        acsf_d_g4[:, 2:].sum(-1) - g[:, 2:].sum(-1))) < 1E-6, text


def test_batch_g4(device):
    """Test G4 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o, cho])
    acsf = Acsf(geo, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()

    acsf_d = ACSF(species=geo.unique_atomic_numbers().numpy(), rcut=rcut,
                  g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = pack([torch.from_numpy(acsf_d.create(ch4)),
                      torch.from_numpy(acsf_d.create(h2o)),
                      torch.from_numpy(acsf_d.create(cho))])

    assert torch.max(abs(acsf_d_g4[..., 2:].sum(-1) -
                         g[..., 2:].sum(-1))) < 1E-6, text


def test_single_g5(device):
    """Test G5 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut,
                g5_params=torch.tensor([[0.02, 1.0, -1.0]]),
                element_resolve=True)
    acsf()

    acsf2 = Acsf(geo, g1_params=rcut,
                 g5_params=torch.tensor([[0.02, 1.0, -1.0]]),
                  element_resolve=True)
    acsf2()

    # get reference from Dscribe
    acsf_d = ACSF(species=species, rcut=rcut, g5_params=[[0.02, 1.0, -1.0]])
    acsf_d_g5 = torch.from_numpy(acsf_d.create(ch4))

    assert torch.max(abs(
        acsf_d_g5[..., 2:].sum(-1) - acsf.g[..., 2:].sum(-1))) < 1E-6, text
    assert torch.max(abs(
        acsf_d_g5[..., 2:].sum(-1) - acsf2.g[..., 2:].sum(-1))) < 1E-6, text


def test_batch_g5(device):
    """Test G5 values in batch geometry."""


def test_batch(device):
    """Test G4 values in batch geometry."""
    rcut = 6.0
    g2_params=torch.tensor([0.5, 1.0])
    g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]])
    geo = Geometry.from_ase_atoms([ch4, h2o])
    acsf = Acsf(geo, g1_params=rcut, g2_params=g2_params,
                g4_params=g4_params, element_resolve=True)
    acsf()

    acsf_d = ACSF(species=geo.unique_atomic_numbers().numpy(), rcut=rcut,
                  g2_params=[[0.5, 1.0]], g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = pack([torch.from_numpy(acsf_d.create(ch4)),
                      torch.from_numpy(acsf_d.create(h2o))])
