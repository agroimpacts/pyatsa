
# %%
from pyatsa import configs
from pyatsa import masking
from skimage.morphology import dilation, opening
import numpy as np
import time
from pyatsa.tseries_preprocess import read_clean_tseries_as_xarr

t_series = read_clean_tseries_as_xarr(
    "/home/rave/pyatsa/cfg/buffered_imgs/ag-forestry/")
configs = configs.ATSA_Configs()

reflectance = np.array(t_series['reflectance'])
angles = np.array(t_series['angles'])

# %%

start = time.time()

hot_t_series, intercepts_slopes = masking.compute_hot_series(
    reflectance, configs.rmin, configs.rmax)

initial_kmeans_clouds, kmeans_centers = masking.sample_and_kmeans(
    hot_t_series, hard_hot=5000, sample_size=10000)

upper_thresh_arr, hot_potential_clear, hot_potential_cloudy = masking.calculate_upper_thresh(
    hot_t_series, initial_kmeans_clouds, configs.A_cloud)

refined_masks = masking.apply_upper_thresh(reflectance, hot_t_series, upper_thresh_arr,
                                           initial_kmeans_clouds, hot_potential_clear, hot_potential_cloudy, configs.dn_max)

# axis 0 must be the image count axis, not height or width
# refined_masks = np.apply_along_axis(opening, 0, refined_masks) # removes single pixel clouds

# refined_masks = np.apply_along_axis(lambda x: dilation(x, selem=np.ones(5,5)), 0, refined_masks)

for i in np.arange(refined_masks.shape[0]):
    refined_masks[i] = opening(refined_masks[i], np.ones((5, 5)))

    # before dilating we need to check for water. where statement currnetly contains hardcoded value to deal with intermittent water being
    # misclassified as cloud due o HOT index not working over water. We can't generate an accurate water mask with Planet alone because
    # it does not have shortwave infrared. see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018RG000598
    refined_masks[i] = np.where((reflectance[i,:, :, 3] < 2000) & (
        refined_masks[i] == 2), 1, refined_masks[i])

    refined_masks[i] = dilation(refined_masks[i], np.ones((5, 5)))

print("seconds ", time.time()-start)
print("finished cloud masking")

# %%
start = time.time()

h_high, h_low = masking.cloud_height_min_max(
    angles, configs.longest_d, configs.shortest_d)
h_ranges = masking.cloud_height_ranges(h_high, h_low)
shift_coords = masking.shadow_shift_coords(h_ranges, angles)
potential_shadow_masks = masking.make_potential_shadow_masks_multi(
    shift_coords, refined_masks)

print("seconds ", time.time()-start)
print("finished potential shadow masking")

# %%
start = time.time()

clearest_land_nir, clearest_land_index = min_cloud_nir(
    potential_shadow_masks, reflectance)
gains_biases = gains_and_biases(
    potential_shadow_masks, reflectance[:,:,:,3], clearest_land_nir, clearest_land_index)
shadow_inds = shadow_index_land(
    potential_shadow_masks, reflectance[:,:,:,3], gains_biases)
li_refined_shadow_masks = masking.apply_li_threshold_multi(
    shadow_inds, potential_shadow_masks)

# 2 is cloud, 1 is clear land, 0 is shadow
cloud_shadow_masks = np.where(li_refined_shadow_masks == 0, 0, refined_masks)

print("seconds ", time.time()-start)
print("finished refined shadow masking")


#%%
import rioxarray
cloud_shadow_masks

#%%
import rioxarray
coords_copy = t_series['reflectance'].drop('band').coords
masks = xr.DataArray(cloud_shadow_masks, dims=['time', 'y', 'x'], coords=[np.ones(44), np.ones(1082), np.ones(1084)])
masks.assign_coords(time=coords_copy['time'], x=coords_copy['x'], y=coords_copy['y'])

#%%
