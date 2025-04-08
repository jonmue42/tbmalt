The data used for this paper is available within this ZIP file.
`dataset` folder contains the geometries and the DFT references for various systems used to train and test models.
`skf` folder contains the Slater-Koster parameter sets.
To reproduce the results shown in the paper, please follow the instruction of the examples in `tbmalt\tests\test\train\` and use the data.

# Description of datasets
`fhi-aims_si63v_hse_101.hdf` and `fhi-aims_si32c31_hse_82.hdf` include geometries of randomly rattled silicon vacancy and silicon carbide with carbon vacancy as well as the corresponding DFT results calculated by FHI-aims with the HSE06 functional, respectively.
They were used to generate the results in Figure 2, Figure 3, Figure 4, Figure 5, Figure 8 and the supporting figures.

`fhi-aims_si63v_relax_pbe.hdf` and `fhi-aims_si512v_pbe.hdf`, `fhi-aims_si65_100_relax_pbe.hdf` and `fhi-aims_si513_100_pbe.hdf`, `fhi-aims_si65_110_relax_pbe.hdf` and `fhi-aims_si513_110_pbe.hdf`, `fhi-aims_si32c31_relax_pbe.hdf` and `fhi-aims_si256c255_pbe.hdf` contain geometries of small and large systems for various materials as well as the corresponding DFT results calculated by FHI-aims with the GGA-PBE functional, respectively.
They were used to generate the results in Figure 7.

`fhi-aims_si65_interstitial_hse.hdf` contains geometries of silicon interstitial defects as well as the corresponding DFT results calculated by FHI-aims with the HSE06 functional.
It was used to generate the results in Figure 6 and the supporting figures.

# Description of skf
`skf_siband.hdf` contains the siband-1-1 Slater-Koster parameter set for the Si-Si pair, with additional five grid points contributing to smooth tails.
`skf_pbc.hdf` contains the pbc-0-3 Slater-Koster parameter set, with additional grid points contributing to smooth tails.