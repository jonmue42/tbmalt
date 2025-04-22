import torch

training_globals = {
        'learning_rate': 0.00005,
        'number_of_epochs': 1,
        'dos_sigma': 0.09,
        }

si63v_hse_101_vars = {
        'batch_size_train': 2,
        'batch_size_test': 2,
        'training_split': [2, 4, 95],
        'points': torch.linspace(-3.0, 2.0, 501),
        'dataset_path': './data_wenbo/dataset/fhi-aims_si63v_hse_101.hdf',
        'parameter_db_path': './data _tbmaltpaper/siband.hdf5',
        }

dataset_vars = {
        'si63v_hse_101': si63v_hse_101_vars
        }
