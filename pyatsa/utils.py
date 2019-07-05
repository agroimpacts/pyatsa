from rasterio.plot import reshape_as_raster

import skimage.io as skio
import numpy as np
import rasterio as rio

path_id = "water"

def save_chips(path_id)
    """
    Run after pyatsa make inputs to save each image as RGB. Make sure paths are setup before running.
    """
    with rio.open('/home/rave/cloud-free-planet/cfg/buffered_stacked/'+path_id+ '_stacked.tif') as dataset:
        profile = dataset.profile.copy()
        meta = dataset.meta.copy()
        arr = dataset.read()

    arr = np.moveaxis(arr, 0, 2)
    arr = np.reshape(arr, (arr.shape[0],arr.shape[1], 4, int(arr.shape[2]/4)), order='F')

    for i in np.arange(arr.shape[-1]):
        blue = arr[:,:,0,i]
        green = arr[:,:,1,i]
        red = arr[:,:,2,i]
        stacked = np.stack([red, green, blue], axis = 2)
        out_path = "/home/rave/cloud-free-planet/cfg/buffered_chips/scene_number_"+str(i)+"_"+path_id+"_.tif"
        profile.update(count=3)
        with rio.open(out_path,'w',**profile) as dst:
            dst.write(reshape_as_raster(stacked))