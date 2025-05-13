import torch

import numpy as np
import matplotlib.pyplot as plt

from tbmalt.physics.dftb.properties import dos
from tbmalt.data.units import energy_units, length_units
from tbmalt import Geometry, OrbitalInfo

from training_vars import training_globals, dataset_vars

def plot_dos_test(dataloader,
                  training_size,
                  batch_size,
                  dftb_calculator,
                  shell_dict,
                  points,
                  labels,
                  title
                  ):

    energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    energies_batch = torch.linspace(-18, 5, 500).repeat(batch_size, 1)

    fermi_ref_tot= torch.empty((0, 1))
    dos_ref_tot = torch.empty((0, 500))
    
    fermi_dftb_tot = torch.empty((0, 1))
    dos_dftb_tot = torch.empty((0, 500))
    for batch, data in enumerate(dataloader):
        targets = {'eigenvalues': data['eigenvalue'],
                   'homo_lumos': data['homo_lumo']
                }
        ref_hl = targets['homo_lumos']
        ref_ev = targets['eigenvalues']
        ref_fermi = targets['homo_lumos'].mean(dim=-1).unsqueeze(-1)
        fermi_ref_tot = torch.cat((fermi_ref_tot, ref_fermi), dim=0)
        ref_dos = dos((ref_ev), energies_batch, training_globals['dos_sigma'])
        dos_ref_tot = torch.cat((dos_ref_tot, ref_dos), dim=0)
        
        geometry_batch = Geometry(data['number'], 
                                 data['position'],
                                 lattice_vector=data['latvec'],
                                 units='a',
                                 cutoff=torch.tensor([18.0])/length_units['angstrom']
                                 )
        orbs_batch = OrbitalInfo(geometry_batch.atomic_numbers, shell_dict, shell_resolved=False)

        dftb_calculator(geometry_batch, orbs_batch, grad_mode='direct')

        hl_dftb = getattr(dftb_calculator, 'homo_lumo').detach() / energy_units['ev']
        fermi_dftb = hl_dftb.mean(-1).unsqueeze(-1)
        fermi_dftb_tot = torch.cat((fermi_dftb_tot, fermi_dftb), dim=0)
        eigval_dftb = dftb_calculator.eig_values.detach() / energy_units['ev']
        dos_dftb = dos((eigval_dftb), energies_batch, training_globals['dos_sigma'])
        dos_dftb_tot = torch.cat((dos_dftb_tot, dos_dftb), dim=0)

    dos_ref_mean = dos_ref_tot.mean(dim=0)
    dos_ref_std = dos_ref_tot.std(dim=0)

    dos_dftb_mean = dos_dftb_tot.mean(dim=0)
    dos_dftb_std = dos_dftb_tot.std(dim=0)
    # Plotting
    plt.plot((energies_plot - fermi_ref_tot)[0], dos_ref_mean, '-', label=labels[0])
    plt.fill_between((energies_plot - fermi_ref_tot)[0],
                     dos_ref_mean + dos_ref_std,
                     dos_ref_mean - dos_ref_std,
                     alpha=0.5)
    plt.plot((energies_plot - fermi_dftb_tot)[0], dos_dftb_mean, '-', label=labels[1])
    plt.fill_between((energies_plot - fermi_dftb_tot)[0],
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
    print('ref fermi:', ref_fermi_plot)
    ref_energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    ref_dos_plot = dos((ref_ev_plot), ref_energies_plot, training_globals['dos_sigma'])
    ref_dos_mean_plot = ref_dos_plot.mean(dim=0)
    ref_dos_std_plot = ref_dos_plot.std(dim=0)

    # Calculator
    dftb_calculator(geometry, orbs)
    
    energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    hl_dftb = getattr(dftb_calculator, 'homo_lumo').detach() / energy_units['ev']
    fermi_dftb = hl_dftb.mean(-1).unsqueeze(-1)
    print('fermi_dftb: ', fermi_dftb)
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
    plt.plot((energies_plot - fermi_dftb)[0], dos_dftb_mean, '-', label=labels[1])
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
     
def plot_interpolation(interpolator, title):
    x_o = interpolator.xp.detach()
    y_o = interpolator.y.detach()
    plt.plot(x_o, y_o, 'o', label='original')
    print('x_o:', x_o.shape)
    #x_test = torch.linspace(0.8, 18, 87)
    #plt.plot(x_test, y_o[:87], 'o', label='original_test')

     
    x = torch.linspace(x_o[0], x_o[-1], 10000)
    #x = torch.linspace(0.8, 18, 10000)
    print('x:', x.shape)
    y_inter = interpolator.forward(x).detach()
    plt.plot(x, y_inter, '.-', label='interpolated')
    plt.title(title)
    plt.legend()

    #y_inter = interpolator.forward(x_o)
    #plt.plot(x_o, y_inter, '-', label='interpolated')

    plt.show()


    
    
        
            
            
