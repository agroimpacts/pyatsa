# pyATSA

pyatsa contains a working package that ports the original ATSA method and code by Zhu and Helmer 2018. Currently the python version produces nearly equivalent results for Planetscope 4-band images that do not contain lots of bright impervious surface, bright soils, or water.

To compare pyatsa to the original IDL version, distributed by Zhu at https://xiaolinzhu.weebly.com/open-source-code.html, it is necessary to have IDL and ENVI installed. then:

  1. Start the idl ide by calling `idlde` in the terminal

  2. Run `ENVI` in the idlde, isnce ENVI functions are required to open the time series file

  3. run `CD, "<path to directory with ATSA-Planet.pro>"`

  4. Compile the ATSA-Planet.pro file with `.r ATSA-Planet.pro`

  5. Call the idl script with `ATSA` from idlde

The IDL script will save out a file that ends in ".sav" (you need to edit the script to change the output path of this file on your computer). It will contain all the variables and results from the script. The file can be read with `scipy.io.readsav` as a python dictionary where each key is the origianl idl variable name. This can then be used in the pytests to check for agreement.

To run tests, navigate to the atsa-python folder and run `pytest test_pyatsa.py`. pytest will check for any functions that contain the word "test" and execute that function, producing times and test results for each test function. pytest fixtures are used to define variables that are shared between tests. These tests currently use two fixtures, one for the configsfor the python version (which are copied from the idl version) and a fixture for the ~3Gb .sav file that contains all the variables from the idl version.
 

#### Notes on the IDL version

The IDL code has separate for loops that operate on water regions and land regions.

The water mask is 0 value where water, 1 where there is not water. The mask values for land classes are as follows.

* 3 - background/SLC errors, missing data
* 2 - cloud (see lines 365 through 378)
* 1 - clear land (see lines 323 through 331, where idl returns 1 or 0 from ge condition)
