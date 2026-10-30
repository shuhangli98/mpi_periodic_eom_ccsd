# Parallel periodic EOM-CCSD

An MPI-parallel implementation of periodic EOM-CCSD for crystalline solids.

## Code author

Shuhang Li (shuhangli98@gmail.com)

## Dependencies

The code has been tested with the following software versions:

* PySCF 2.10.0
  * Aug 25, 2025
  * Commit: `6f9d9399cac057e5965e8250cf8e3ef57716510d`
* Python 3.10.13
* NumPy 1.26.4
* SciPy 1.13.0
* mpi4py 3.1.5
* psutil 5.9.8

## Basis sets

This repository includes several basis set files used in the calculations. These files were obtained from the supporting data repository of Hong-Zhou Ye:

https://github.com/hongzhouye/supporting_data

We thank Hong-Zhou Ye for making these data publicly available.

## Reference

If you use this code, please cite:

Shuhang Li, Huanchen Zhai, Francesco A. Evangelista, and Timothy C. Berkelbach,
“Reaching the thermodynamic limit of periodic CCSD cohesive energies and band gaps,”
npj Comput. Mater. [Paper](https://doi.org/10.1038/s41524-026-02372-8), 2026.
