import torch
from tbmalt.data.units import energy_units, length_units

training_globals = {
        'learning_rate': 0.00005,
        #'learning_rate': 0.00005,
        #'learning_rate': 5e-6,
        #'learning_rate': 0.0,
        'number_of_epochs': 1,
        'dos_sigma': 0.09,
        }

si63v_hse_101_vars = {
        'batch_size_train': 1,
        'batch_size_test': 1,
        'training_split': [2, 1, 101-3], #training, testing, throwaway
        'points': torch.linspace(-3.0, 2.0, 501),
        'cutoff': torch.tensor([18.0])/length_units['angstrom'],
        'dataset_path': './data_wenbo/dataset/fhi-aims_si63v_hse_101.hdf',
        'parameter_db_path': './data_tbmaltpaper/siband.hdf5',
        }

si32c31_hse_82_vars = {
        'batch_size_train': 1,
        'batch_size_test': 1,
        'training_split': [3, 3, 82-6], #training, testing, throwaway
        'points': torch.linspace(-4.1, 0.2, 431),
        'cutoff': torch.tensor([9.98])/length_units['angstrom'],
        'dataset_path': './data_wenbo/dataset/fhi-aims_si32c31_hse_82.hdf',
        'parameter_db_path': './output.hdf5',
        }

dataset_vars = {
        'si63v_hse_101': si63v_hse_101_vars,
        'si32c31_hse_82': si32c31_hse_82_vars,
        }
