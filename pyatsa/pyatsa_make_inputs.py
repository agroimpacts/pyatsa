import numpy as np
import rasterio as rio
import skimage as ski
import glob
import os
from rasterio.plot import reshape_as_raster
import json

def keep_imgs_with_meta(img_paths, meta_paths, udm_paths, udm2_paths, DIR):
    imgs_match = []
    meta_match = []
    udm_match = []
    udm2_match = []
    for im in list(map(os.path.basename, sorted(img_paths))):
        for meta in list(map(os.path.basename, sorted(meta_paths))):
            for udm in list(map(os.path.basename, sorted(udm_paths))):
                for udm2 in list(map(os.path.basename, sorted(udm2_paths))):
                    if meta[0:15] == im[0:15] == udm[0:15] == udm2[0:15]:
                        imgs_match.append(im)
                        meta_match.append(meta)
                        udm_match.append(udm)
                        udm2_match.append(udm2)
    imgs_match = [os.path.join(DIR,name) for name in imgs_match]
    meta_match = [os.path.join(DIR,name) for name in meta_match]
    udm_match = [os.path.join(DIR,name) for name in udm_match]
    udm2_match = [os.path.join(DIR,name) for name in udm2_match]

    return imgs_match, meta_match, udm_match, udm2_match

def get_sun_elevation_azimuth(path):
    with open(path) as f:
        metadata = json.load(f)
    return (metadata['properties']['sun_elevation'], metadata['properties']['sun_azimuth'])

def save_blank_water_mask(path):
    """Used for testing ATSA where we know there is no water
    Use MOD44W water mask product or something else for a water 
    mask instead outside of testing.
    """
    test = rio.open(path)
    meta = test.profile
    meta.update(count=1) # update since we are writing a single band
    b1_array, b2_array, b3_array, b4_array = test.read()
    fake_mask = np.ones(b1_array.shape)
    with rio.open('fake_mask_nowater.tif', 'w', **meta) as dst:
        dst.write(fake_mask.astype('uint16'), 1)

def stack_t_series(paths, stackname):
    """"
    Stack third axis-wise. all 
    tifs must be same extent and in sorted order by date
    """
    arrs = [ski.io.imread(path) for path in paths]
    stacked = reshape_as_raster(np.dstack(arrs))
    img = rio.open(paths[0])
    meta=img.profile
    if len(arrs[0].shape) == 2:
        meta.update(count=len(arrs))
    else:
        meta.update(count=len(arrs)*arrs[0].shape[2])
    with rio.open(stackname, 'w',  **meta) as dst:
        dst.write(stacked)
    if len(arrs[0].shape) == 2:
        print("Saved Time Series with " + str(len(arrs)) + " images and " + str(1) + " bands each")
    else:
        print("Saved Time Series with " + str(len(arrs)) + " images and " + str(arrs[0].shape[2]) + " bands each")

if __name__ == '__main__':
    path_id = "savanna"
    DIR = os.path.join("/home/rave/cloud-free-planet/cfg/buffered_imgs/",path_id)
    sr_pattern = "/home/rave/cloud-free-planet/cfg/buffered_imgs/"+path_id+"/*SR*.tif"
    img_paths = glob.glob(sr_pattern)
    img_paths = sorted(img_paths)
    meta_pattern = "/home/rave/cloud-free-planet/cfg/buffered_imgs/"+path_id+"/*meta*.json"
    meta_paths = glob.glob(meta_pattern)
    meta_paths = sorted(meta_paths)
    udm_pattern = "/home/rave/cloud-free-planet/cfg/buffered_imgs/"+path_id+"/*udm_*.tif"
    udm_paths = glob.glob(udm_pattern)
    udm_paths = sorted(udm_paths)
    udm2_pattern = "/home/rave/cloud-free-planet/cfg/buffered_imgs/"+path_id+"/*udm2*.tif"
    udm2_paths = glob.glob(udm2_pattern)
    udm2_paths = sorted(udm2_paths)
    imgs_match, meta_match, udm_match, udm2_match = keep_imgs_with_meta(img_paths, meta_paths, udm_paths, udm2_paths, DIR)
    
    angles = list(map(get_sun_elevation_azimuth, meta_match))
    with open(os.path.join("/home/rave/cloud-free-planet/cfg/buffered_angles", path_id+'_angles_larger_utm.txt'), 'w') as f:
        for tup in angles:
            f.write('%s %s\n' % tup)
    
    save_blank_water_mask(imgs_match[0])

    stack_t_series(imgs_match, os.path.join("/home/rave/cloud-free-planet/cfg/buffered_stacked", path_id+"_stacked.tif"))
    stack_t_series(udm_match, os.path.join("/home/rave/cloud-free-planet/cfg/buffered_stacked", path_id+"_udm_stacked.tif"))
    stack_t_series(udm2_match, os.path.join("/home/rave/cloud-free-planet/cfg/buffered_stacked", path_id+"_udm2_stacked.tif"))

