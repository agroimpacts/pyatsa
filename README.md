# pyATSA
pyatsa finds clouds in satellite imagery.

This repo contains the pyatsa package that ports the original ATSA method and code by Zhu and Helmer 2018, as well as example notebooks. Currently the python version produces nearly equivalent results to Planet Lab's Usable Data Mask for Planetscope 4-band images across Ghana. In cases where there are lots of bright impervious surface, bright soils, or water, pyatsa generates substantial false positives in the cloud shadow class and speckles of false positives from clouds. These are the primary issues being worked on.

# Install

`pip install pyatsa`

## Install from source with dependencies

```
conda env create -f environment.yaml
conda activate pyatsa
# or anaconda3 if you use anaconda or whatever the path to your python file is
flit install --symlink --python ~/miniconda3/envs/pyatsa/bin/python3.7
```

## Differences between pyatsa and original ATSA by Zhu and Helmer 2018

- When computing the Clear Sky Lines for images in a time series, Zhu set A_cloud within .5-1.5 idea, but low values lead to too many false positives in Planetscope imagery.
- I chose to use Li's cross entrpy thresholding method in scikit-image rather than Inverse Distance Weighting to segment cloud shadows within the potential cloud shadow zone calculated from the scene geometry. This was simpler to code and appeared to give better results, the IDW method had a strong bias toward only masking shadows directly adjacent to clouds. I observed cloud shadows that were almost a kilometer away from their cloud shadows throughout Ghana, where pyatsa is being tested. More actual shadows are correctly masked, though more false positives are generated where clouds are not present.


# Comparing with the original ATSA IDL code
The original ATSA program is distributed by the authors at https://xiaolinzhu.weebly.com/open-source-code.html. To run, it is necessary to have IDL and ENVI installed. then:

  1. Start the idl ide by calling `idlde` in the terminal

  2. Run `ENVI` in the idlde, isnce ENVI functions are required to open the time series file

  3. run `CD, "<path to directory with ATSA-Planet.pro>"`

  4. Compile the ATSA-Planet.pro file with `.r ATSA-Planet.pro`

  5. Call the idl script with `ATSA` from idlde

The IDL script will save out a file that ends in ".sav" (you need to edit the script to change the output path of this file on your computer). It will contain all the variables and results from the script. The file can be read with `scipy.io.readsav` as a python dictionary where each key is the origianl idl variable name. This can then be used in the pytests to check for agreement.

To run tests, navigate to the atsa-python folder and run `pytest test_pyatsa.py`. pytest will check for any functions that contain the word "test" and execute that function, producing times and test results for each test function. pytest fixtures are used to define variables that are shared between tests. These tests currently use two fixtures, one for the configsfor the python version (which are copied from the idl version) and a fixture for the ~3Gb .sav file that contains all the variables from the idl version.

#### Notes on the IDL version to assist with debugging

The IDL code has separate for loops that operate on water regions (identified by a pre-existing water mask) and land regions.

The water mask is 0 value where water, 1 where there is not water. The mask values for land classes are as follows.

* 3 - background/SLC errors, missing data
* 2 - cloud (see lines 365 through 378)
* 1 - clear land (see lines 323 through 331, where idl returns 1 or 0 from ge condition)
