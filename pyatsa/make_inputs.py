import numpy as np
import rasterio as rio
import skimage as ski
from glob import glob
import os
from rasterio.plot import reshape_as_raster
import json
import xarray as xr
import rioxarray
from datetime import datetime

def get_sun_elevation_azimuth(path):
    with open(path) as f:
        metadata = json.load(f)
    return np.array([metadata['properties']['sun_elevation'], metadata['properties']['sun_azimuth']])


def save_blank_water_mask(path):
    """Used for testing ATSA where we know there is no water
    Use MOD44W water mask product or something else for a water 
    mask instead outside of testing.
    """
    test = rio.open(path)
    meta = test.profile
    meta.update(count=1)  # update since we are writing a single band
    b1_array, b2_array, b3_array, b4_array = test.read()
    fake_mask = np.ones(b1_array.shape)
    with rio.open('fake_mask_nowater.tif', 'w', **meta) as dst:
        dst.write(fake_mask.astype('uint16'), 1)


def open_rasterio_planet(path, chunk_coord, chunk_size):
    """Reads in a Planet Analytic 4-band surface reflectance image
    and correctly assigns the band metadata. Also works with single

    Args:
        path (str): Path of form
            .../20180923_103115_0f46_3B_AnalyticMS_DN.tif. Or udm/udm2.tif
        chunk_coord (str): the coordinate to chunk along. chunking across time
            is probably best since kmeans and thresholding are performed on a
            per scene basis and take a while. To do this, set chunk size equal
            to the number of bands and chunk_coord to 'band'.
        chunk_size (int): The size of the chunk.

    Returns:
        bool: Returns an xarray data array with 3 dimensions (band, y, x)
    """
    data_array = xr.open_rasterio(
        path, chunks={chunk_coord: chunk_size})  # chunks makes it lazily executed
    return data_array


def write_xarray_lsr(xr_arr, fpath):
    xr_arr.rio.to_raster(fpath)


def read_scenes(path_pattern, chunk_coord, chunk_size):
    """
    Reads in multiple Planetscope scenes given a wildcard pattern.

    Args:
        path (str): Path of form "../*".
        chunk_coord (str): the coordinate to chunk along. chunking across time
            is probably best since kmeans and thresholding are performed on a
            per scene basis and take a while. To do this, set chunk size equal
            to the number of bands and chunk_coord to 'band'.
        chunk_size (int): The size of the chunk.

    Returns:
        bool: Returns an xarray data array with 4 dimensions (time, band, y, x)
            the time is a datetime coordinate and can be indexed like 
            x.sel(slice('2019-01-01', '2019-02-01'))
    """
    scenes = glob(path_pattern)
    scenes.sort(key=lambda path: datetime.strptime(os.path.basename(path)[0:8],
                                                   "%Y%m%d"))
    xr_arrs = [open_rasterio_planet(
        path, chunk_coord, chunk_size) for path in scenes]
    t_series_xarr = xr.concat(xr_arrs, dim="time")
    sorted_dates = [datetime.strptime(os.path.basename(
        path)[0:8], "%Y%m%d") for path in scenes]
    return t_series_xarr.assign_coords(time=sorted_dates)


def read_angles(path_pattern):
    """
    Reads in multiple Planetscope scenes' metadata and returns an xarray of angles

    Args:
        path_pattern (str): Path of form "../*metadata.json".

    Returns:
        bool: Returns an xarray data array with 2 dimensions (time, angle)
            the time is a datetime coordinate and can be indexed like 
            x.sel(slice('2019-01-01', '2019-02-01'))
    """
    angles = glob(path_pattern)
    angles.sort(key=lambda path: datetime.strptime(os.path.basename(path)[0:8],
                                                   "%Y%m%d"))
    sorted_dates = [datetime.strptime(os.path.basename(
        path)[0:8], "%Y%m%d") for path in angles]
    angle_data = np.array([get_sun_elevation_azimuth(angle_path) for angle_path in angles])
    labels = ['sun_elev','azimuth']
    return xr.DataArray(angle_data, dims=['time', 'angle'], coords = [sorted_dates, labels])