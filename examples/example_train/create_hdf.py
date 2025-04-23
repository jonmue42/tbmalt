import time
from os.path import join
import urllib, tempfile, tarfile
import torch
from tbmalt.io.skf import Skf
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

#input_file = "./data_wenbo/skf/skf_pbc.hdf"
##
#atoms = ['C', 'H', 'N']
#def add_dash(old_key):
#    if old_key[0] in atoms:
#        return old_key[0] + '-' + old_key[1:]
#    else:
#        return old_key[0:2] + '-' + old_key[2:]
#
#
#with h5py.File(input_file, 'r+') as f:
#    for key in f.keys():
#        f.move(key, add_dash(key))
#    print(f.keys())
#
#print(add_dash('CC'))
#print(add_dash('CSi'))
#print(add_dash('SiSi'))
#

def skf_file(output_path: str):
    """Path to auorg-1-1 HDF5 database.

    This function downloads the auorg-1-1 Slater-Koster parameter set & converts
    it to HDF5 database stored at the path provided.

    Arguments:
         output_path: location to where the auorg-1-1 HDF5 database file should
            be stored.

    Warnings:
        This will fail without an internet connection.

    """
    # Link to the auorg-1-1 parameter set
    #link = 'https://dftb.org/fileadmin/DFTB/public/slako/auorg/auorg-1-1.tar.xz'
    link = 'https://github.com/dftbparams/pbc/releases/download/v0.3.0/pbc-0-3.tar.xz'

    # Elements of interest
    elements = ['H', 'C', 'N', 'Si']

    with tempfile.TemporaryDirectory() as tmpdir:

        # Download and extract the auorg parameter set to the temporary directory
        urllib.request.urlretrieve(link, path := join(tmpdir, 'pbc-0-3.tar.xz'))
        with tarfile.open(path) as tar:
            tar.extractall(tmpdir)

        # Select the relevant skf files and place them into an HDF5 database
        skf_files = [join(tmpdir, 'pbc-0-3', f'{i}-{j}.skf')
                     for i in elements for j in elements]

        for skf_file in skf_files:
            Skf.read(skf_file).write(output_path)

skf_file('output.hdf5')
