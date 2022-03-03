# xforecasting - A toolbox for spatio-temporal forecasting with xarray and pytorch.

The code in this repository provides a scalable and flexible framework to develop spatio-temporal forecasting models. 
It princiapially builds upon xarray, pytorch and zarr libraries.

ATTENTION: The code is subject to changes in the coming months.

The folder `tutorials` (will) provide jupyter notebooks describing various features of x-forecasting.

The folder `docs` (will) contains slides and notebooks explaining the xforecasting framework.

## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/ghiggi/xforecasting.git
   cd x-forecasting
   ```
  
2. Install manually the following dependencies:
   - Install first pytorch and its extensions on GPU:
      ```sh
      conda install -c conda-forge pytorch-gpu  
      ```
   - If you don't have GPU available install it on CPU:
      ```sh
      conda install -c conda-forge pytorch-cpu  
      ```
   - Install the other required packages: 
   ```sh
   conda create --name xforecasting-dev python=3.8
   conda install -c conda-forge xarray dask cdo h5py h5netcdf netcdf4 zarr numcodecs rechunker
   conda install -c conda-forge notebook jupyterlab
   conda install -c conda-forge matplotlib-base cycler
   conda install -c conda-forge numpy pandas numba scipy bottleneck tabulate
   ```
   
2. Alternatively install the dependencies using one of the appropriate below 
   environment.yml files:
   ```sh
   conda env create -f TODO.yml
   ```

## Tutorials

## Reproducing our results

## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
* [Yann Yasser Haddad](https://www.linkedin.com/in/yann-yasser-haddad)

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
