import torch

import numpy as np
import matplotlib.pyplot as plt

from tbmalt.physics.dftb.properties import dos
from tbmalt.data.units import energy_units, length_units

from training_vars import training_globals, dataset_vars

def plot_dos(targets,
             training_size,
             geometry,
             orbs,
             dftb_calculator,
             points,
             labels,
             title
             ):
    
    #Reference
    ref_hl_plot = targets['homo_lumos']
    ref_ev_plot = targets['eigenvalues']
    ref_fermi_plot = targets['homo_lumos'].mean(dim=-1).unsqueeze(-1)
    ref_energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    ref_dos_plot = dos((ref_ev_plot), ref_energies_plot, training_globals['dos_sigma'])
    ref_dos_mean_plot = ref_dos_plot.mean(dim=0)
    ref_dos_std_plot = ref_dos_plot.std(dim=0)

    # Calculator
    dftb_calculator(geometry, orbs)
    
    energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    hl_dftb = getattr(dftb_calculator, 'homo_lumo').detach() / energy_units['ev']
    fermi_dftb = hl_dftb.mean(-1).unsqueeze(-1)
    eigval_dftb = dftb_calculator.eig_values.detach() / energy_units['ev']
    dos_dftb = dos((eigval_dftb), energies_plot, training_globals['dos_sigma'])
    dos_dftb_mean = dos_dftb.mean(dim=0)
    dos_dftb_std = dos_dftb.std(dim=0)

    # Plotting
    plt.plot((ref_energies_plot - ref_fermi_plot)[0], ref_dos_mean_plot, '-', label=labels[0])
    plt.fill_between((ref_energies_plot - ref_fermi_plot)[0],
                     ref_dos_mean_plot + ref_dos_std_plot,
                     ref_dos_mean_plot - ref_dos_std_plot,
                     alpha=0.5)
    plt.plot((ref_energies_plot - fermi_dftb)[0], dos_dftb_mean, '-', label=labels[1])
    plt.fill_between((ref_energies_plot - fermi_dftb)[0],
                     dos_dftb_mean + dos_dftb_std,
                     dos_dftb_mean - dos_dftb_std,
                     alpha=0.5)
    
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on', right='on')
    plt.xlim((points[0], points[-1]))
    
    plt.xlabel(r'E - $\mathregular{E_f}$ [eV]', fontsize=15)
    plt.ylabel('DOS [states / eV]', fontsize=15)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=13)
    plt.show()

def plot_training_ref(targets, training_size, points):
    #Reference
    ref_hl_plot = targets['homo_lumos']
    ref_ev_plot = targets['eigenvalues']
    ref_fermi_plot = targets['homo_lumos'].mean(dim=-1).unsqueeze(-1)
    ref_energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    ref_dos_plot = dos((ref_ev_plot), ref_energies_plot, training_globals['dos_sigma'])
    ref_dos_mean_plot = ref_dos_plot.mean(dim=0)
    ref_dos_std_plot = ref_dos_plot.std(dim=0)
    
    plt.plot((ref_energies_plot - ref_fermi_plot)[0], ref_dos_mean_plot, '-')
    plt.fill_between((ref_energies_plot - ref_fermi_plot)[0],
                     ref_dos_mean_plot + ref_dos_std_plot,
                     ref_dos_mean_plot - ref_dos_std_plot,
                     alpha=0.5)
    plt.fill_between(points, -3, 80, alpha=0.2)

    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on', right='on', zorder=10)
    plt.xlim((-4, 7))
    #plt.ylim((-1, 70))
    plt.xlabel(r'E - $\mathregular{E_f}$ [eV]', fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.show()
     

    
    
        
            
            
