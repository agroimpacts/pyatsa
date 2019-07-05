import numpy as np
import rasterio as rio
from rasterio import fill
import skimage as ski
import matplotlib.pyplot as plt
import glob
import os
from rasterio.plot import reshape_as_raster, reshape_as_image
import json
import scipy.stats as stats
from scipy import io
import statsmodels.formula.api
from skimage import exposure
from sklearn.cluster import KMeans
from skimage import morphology as morph
from math import ceil
import pyatsa_configs
import pyatsa
import pytest

@pytest.fixture
def configs():
    ATSA_DIR="/home/rave/cloud-free-planet/atsa-test-unzipped/"
    img_path = os.path.join(ATSA_DIR, "planet-pyatsa-test/stacked_larger_utm.tif")
    return pyatsa_configs.ATSA_Configs(img_path, ATSA_DIR)

@pytest.fixture
def idl_variables_results():
    return io.readsav("atsa-idl-variables.sav")

def test_angles():
    return np.allclose(configs.h_high.reshape(idl_variables_results['h_high'].shape), idl_variables_results['h_high'])

def test_t_series_shape(configs):
    assert len(configs.t_series.shape) == 4 # make sure t_series is 4D
    assert configs.t_series.shape[-1] > 1 # make sure t_series has more than 1 image

def test_histo_labels(idl_variables_results, configs):
    raise ValueError

def test_get_bin_means(idl_variables_results, configs):
    raise ValueError

def test_intercept_and_slope(idl_variables_results, configs):
    raise ValueError

def test_get_clear_skyline(idl_variables_results, configs):
    raise ValueError

def test_compute_hot_series(idl_variables_results, configs):
    raise ValueError

def test_sample_and_kmeans(idl_variables_results, configs):
    raise ValueError

def test_calculate_upper_thresh(idl_variables_results, configs):
    raise ValueError