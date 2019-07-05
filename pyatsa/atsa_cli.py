import numpy as np
import rasterio as rio
import skimage as ski
import glob
import os
import click
from rasterio.plot import reshape_as_raster

@click.group()
def cli():
    pass

@click.command()
@click.argument('--path', help='path to the ')
def get_sun_elevation_azimuth(path):
    with open(path) as f:
        metadata = json.load(f)
    return (metadata['properties']['sun_elevation'], metadata['properties']['sun_azimuth'])

@click.command()
def save_blank_water_mask(path):
    """Used for testing ATSA where we know ther eis no water
    In the future we can use MOD44W water mask product or something else
    """
    test = rio.open(path)
    meta = test.profile
    meta.update(count=1) # update since we are writing a single band
    b1_array, b2_array, b3_array, b4_array = test.read()
    fake_mask = np.zeros(b1_array.shape)
    with rio.open('fake_mask.tif', 'w', **meta) as dst:
        dst.write(fake_mask.astype('uint16'), 1)

@click.command()
def stack_t_series(paths, stackname):
    """"
    Stack third axis-wise. all
    tifs must be same extent and in sorted order by date
    """
    arrs = [ski.io.imread(path) for path in paths]
    stacked = reshape_as_raster(np.dstack(arrs))
    img = rio.open(paths[0])
    meta=img.profile
    meta.update(count=len(arrs)*arrs[0].shape[2])
    with rio.open(stackname, 'w',  **meta) as dst:
        dst.write(stacked)
    print("Saved Time Series with " + str(len(arrs)) + " images and " + str(arrs[0].shape[2]) + " bands each")

cli.add_command(get_sun_elevation_azimuth)
cli.add_command(save_blank_water_mask)
cli.add_command(stack_t_series)

if __name__ == "__main__":


    sr_pattern = "/home/rave/cloud-free-planet/notebooks/jan_april_2018_100ovp_50maxcloud/*SR*.tif"
    img_paths = glob.glob(sr_pattern)

    img_paths = sorted(img_paths)

    meta_pattern = "/home/rave/cloud-free-planet/notebooks/jan-may/*metadata.json"
    meta_paths = glob.glob(meta_pattern)

    angles = list(map(get_sun_elevation_azimuth, meta_paths))
    with open('angles.txt', 'w') as f:
        for tup in angles:
            f.write('%s %s\n' % tup)

    save_blank_water_mask(img_paths[0])

    stack_t_series(img_paths, "stacked.tif")
    cli()
