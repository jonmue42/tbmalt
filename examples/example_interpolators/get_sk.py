import os
from os.path import join
import urllib, tempfile, tarfile, zipfile

import torch

from tbmalt.io.skf import Skf
torch.set_default_dtype(torch.float64)

# Link to the auorg-1-1 parameter set
link = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'

# Elements of interest
elements = ['H', 'C', 'N', 'O', 'S', 'Au']

# Path of the parameter set
try:
    os.mkdir("./data")
except FileExistsError:
    pass
    
parameter_db_path = "./data/auorg.h5"

with tempfile.TemporaryDirectory() as tmpdir:

    # Download and extract the auorg parameter set to the temporary directory
    urllib.request.urlretrieve(link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
    with tarfile.open(path) as tar:
        tar.extractall(tmpdir)

    # Select the relevant skf files and place them into an HDF5 database
    skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
                 for i in elements for j in elements]

    for skf_file in skf_files:
        Skf.read(skf_file).write(parameter_db_path)


