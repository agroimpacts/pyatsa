import numpy as np
import rasterio as rio
import skimage as ski
from glob import glob
import os
import json
import xarray as xr
import rioxarray
from datetime import datetime
from collections import Counter


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
    band udm or udm2

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


def read_scenes(scenes, chunk_coord, chunk_size, remove_list=[]):
    """
    Reads in multiple Planetscope scenes given a wildcard pattern.

    Args:
        scenes (str): Paths of form "../*".
        chunk_coord (str): the coordinate to chunk along. chunking across time
            is probably best since kmeans and thresholding are performed on a
            per scene basis and take a while. To do this, set chunk size equal
            to the number of bands and chunk_coord to 'band'.
        chunk_size (int): The size of the chunk.
        remove_list: list of id to remove. ids are a date followed by a cubesat id
            like 20180101_100517.

    Returns:
        bool: Returns an xarray data array with 4 dimensions (time, band, y, x)
            the time is a datetime coordinate and can be indexed like
            x.sel(slice('2019-01-01', '2019-02-01'))
    """
    scenes.sort(key=lambda path: datetime.strptime(os.path.basename(path)[0:8],
                                                   "%Y%m%d"))
    if remove_list:
        for bad in remove_list:
            for scene in scenes:
                if bad == os.path.basename(scene)[0:15]:
                    scenes.remove(scene)
    xr_arrs = [open_rasterio_planet(
        path, chunk_coord, chunk_size) for path in scenes]
    t_series_xarr = xr.concat(xr_arrs, dim="time")
    sorted_dates = [datetime.strptime(os.path.basename(
        path)[0:8], "%Y%m%d") for path in scenes]
    return t_series_xarr.assign_coords(time=sorted_dates)


def remove_bad_paths(paths, remove_list=[]):
    if remove_list:
        for bad in remove_list:
            for path in paths:
                if bad == os.path.basename(path)[0:15]:
                    print("Removed: " + bad)
                    paths.remove(path)
    return paths


def read_angles(angles, remove_list=[]):
    """
    Reads in multiple Planetscope scenes' metadata and returns an xarray of angles

    Args:
        path_pattern (str): Path of form "../*metadata.json".

    Returns:
        bool: Returns an xarray data array with 2 dimensions (time, angle)
            the time is a datetime coordinate and can be indexed like
            x.sel(slice('2019-01-01', '2019-02-01'))
    """
    angles.sort(key=lambda path: datetime.strptime(os.path.basename(path)[0:8],
                                                   "%Y%m%d"))
    angles = remove_bad_paths(angles, remove_list)
    sorted_dates = [datetime.strptime(os.path.basename(
        path)[0:8], "%Y%m%d") for path in angles]
    angle_data = np.array([get_sun_elevation_azimuth(angle_path)
                           for angle_path in angles])
    labels = ['sun_elev', 'azimuth']
    return xr.DataArray(angle_data, dims=['time', 'angle'], coords=[sorted_dates, labels])


def sort_paths_by_date(path_pattern):
    paths = glob(path_pattern)
    paths.sort(key=lambda path: datetime.strptime(os.path.basename(path)[0:8],
                                                  "%Y%m%d"))
    return paths


def sorted_dates_from_paths(paths):
    return [os.path.basename(
        path)[0:8] for path in paths]


def find_duplicates(scene_paths, count=2):
    """
    Finds duplicates in a list given by the path pattern. It checks for matches
    of the date string in the paths and returns the paths that have duplicates.

    path_pattern (str): pattern match to get rid of duplicates.
    count (int), optional: set to find either duplicates, triplicates, etc.
    """

    def get_dupe_dates(dates):
        """
        Dates and paths is a list of tuples of form
        (date string, path string)

        Returns list of paths that have duplicate dates
        """
        return [k for k, v in Counter(dates).items() if v > 1]
    dates = sorted_dates_from_paths(scene_paths)
    duplicate_dates = get_dupe_dates(dates)

    duplicate_paths = []

    for dup in duplicate_dates:
        store = []
        for scene in scene_paths:
            if dup in os.path.basename(scene)[0:8]:
                store.append(scene)
        duplicate_paths.append(store)
        store = []

    return duplicate_paths


def filter_path_list_by_date(path_list, id_list):
    result = []
    for good_id in id_list:
        for path in path_list:
            if os.path.basename(path)[0:8] == good_id and path not in result:
                result.append(path)
    return result


def pick_most_cloudy(path1, path2):
    xarr1 = open_rasterio_planet(path1, chunk_coord='band', chunk_size=1)
    xarr2 = open_rasterio_planet(path2, chunk_coord='band', chunk_size=1)
    # creates masks of clouds and also all quality issues with each of the 4 bands
    if int(xarr1.where(xarr1 != 0).count()) < int(xarr2.where(xarr2 != 0).count()):
        return path2
    else:
        return path1


def get_most_cloudy_ids(duplicate_list):
    """
    Args:
        path_pattern (str): pattern to find files through which to search for duplicates

    Returns:
        list: A list of path duplicates, the ones that are less cloudy.
    """
    most_cloudy_id_list = []
    for dup in duplicate_list:
        most_id = os.path.basename(pick_most_cloudy(*dup))[0:15]
        most_cloudy_id_list.append(most_id)
    return most_cloudy_id_list


def get_meta_ids_to_remove(duplicate_list_meta):
    remove_id_list_meta = []
    for i in duplicate_list_meta:
        remove_id_list_meta.append(os.path.basename(i[1])[0:15])
    return remove_id_list_meta


def read_clean_tseries_as_xarr(input_folder):
    """
    Reads an xarray with all the inputs used by pyatsa

    This will put surface reflectance, udm, udm2, and angle metadata
    into an organized xarray Dataset as variables, where each is aligned
    along the same time coordinate.

    Args:
        input_folder (str): This should be an absolute path to a older that contains
    the downloaded products and metadata from a porder order. Currently only works
    if you download udm, udm2, and metadata alon with 4 band surface reflectance into
    this same folder. TODO, remove udm as a requirement since udm2's 8th band contains
    the udm. (need to double check this is actually the case but the product specs says 
    this)

    TODO need to add the water mask in here as another arg and input into the xarr
    TODO need to make pattern filters an arg?
    Returns:
        xarray.Dataset
    """
    assert input_folder[-1] == '/'
    # selecting patterns to use for detection
    udm_pattern = input_folder + "*udm_clip.tif"
    udm2_pattern = input_folder + "*udm2_clip.tif"
    sr_pattern = input_folder + "*SR_clip.tif"
    meta_pattern = input_folder + "*metadata.json"

    all_udms = sort_paths_by_date(udm_pattern)
    all_udm2s = sort_paths_by_date(udm2_pattern)
    all_srs = sort_paths_by_date(sr_pattern)
    all_metas = sort_paths_by_date(meta_pattern)

    # during download you may not get one to one match between metadata and products
    # so we only keep scnees that have all products and metadta
    id_list = list(set(sorted_dates_from_paths(all_udms)).intersection(
        set(sorted_dates_from_paths(all_udm2s)),
        set(sorted_dates_from_paths(all_srs)),
        set(sorted_dates_from_paths(all_metas))))

    good_udms = filter_path_list_by_date(all_udms, id_list)
    good_udm2s = filter_path_list_by_date(all_udm2s, id_list)
    good_srs = filter_path_list_by_date(all_srs, id_list)
    good_metas = filter_path_list_by_date(all_metas, id_list)

    # remove cloudy duplicate dates
    duplicate_list_udm = find_duplicates(good_udms)
    remove_id_list_cloudy = get_most_cloudy_ids(duplicate_list_udm)

    # remove duplicates in other products and meta, these can be unique to the type of file
    duplicate_list_meta = find_duplicates(good_metas)
    remove_id_list_meta = get_meta_ids_to_remove(duplicate_list_meta)
    duplicate_list_udm2 = find_duplicates(good_udm2s)
    remove_id_list_udm2 = get_meta_ids_to_remove(duplicate_list_udm2)
    duplicate_list_sr = find_duplicates(good_srs)
    remove_id_list_sr = get_meta_ids_to_remove(duplicate_list_sr)

    remove_id_list_all = list(set(remove_id_list_cloudy).union(
        set(remove_id_list_meta),
        set(remove_id_list_sr),
        set(remove_id_list_udm2)))

    # read in the scenes
    udm = read_scenes(good_udms, 'band', 1, remove_id_list_all)
    udm2 = read_scenes(good_udm2s, 'band', 8, remove_id_list_all)
    sr = read_scenes(good_srs, 'band', 4, remove_id_list_all)
    sr = sr.transpose('time', 'y', 'x', 'band')
    angles = read_angles(good_metas, remove_id_list_all)
    return xr.Dataset({'reflectance': sr,
                    'udm': udm,
                    'udm2': udm2,
                    'angles': angles})
