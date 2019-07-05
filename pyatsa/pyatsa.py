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
import statsmodels.formula.api
from sklearn.cluster import KMeans
from math import ceil
from skimage.draw import line
from skimage.morphology import dilation, opening
from skimage.filters import threshold_li
import skimage.io as skio
os.chdir("/home/rave/cloud-free-planet/atsa-python")
import pyatsa_configs
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, cpu_count

def map_processes(func, args_list):
    """
    Set MAX_PROCESSES in preprocess_config.yaml
    args_sequence is a list of lists of args
    """
    processes = cpu_count()-1
    pool = Pool(processes)
    results = pool.starmap(func, args_list)
    pool.close()
    pool.join()
    return results

#Computing the Clear Sky Line for Planet Images in T Series
#Zhu set to 1.5 if it was less than 1.5 but this might not be a good idea for Planet 
#due to poorer calibration?
def reject_outliers_by_med(data, m = 2.):
    """
    Reject outliers based on median deviation
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m].flatten()

def reject_outliers_by_mean(data_red, data_blue, m = 3.):
    """
    Reject outliers based on deviation from mean
    This is the method used in Zhu and Elmer 2018
    """
    return (data_red[data_red <= np.mean(data_red) + m * np.std(data_red)], \
            data_blue[data_red <= np.mean(data_red) + m * np.std(data_red)])

def get_histo_labels(img, rmin0, rmax, nbins=50):
    """
    Takes an image of shape [H, W, Channel], gets blue and red bands,
    and computes histogram for blue values. then it finds the array indices
    for each bin of the histogram.
    
    Args:
        img (numpy array): the input image, part of a time series of images
        rmin0 (int): minimum edge of the histogram (later adjusted based on each image)
        rmax (int): maximum edge of the histogram
        nbins (int): number of histogram bins
        
    Returns:
        the array of histogram indices of same shape as a band or (nan, nan)
        which is later passed to compute hot when nanmean is called.
    
    """
    # make 3D arrays for blue and red bands to compute clear sky lines
    blue = img[:,:,0]
    # finding samples, there should be at least 500 values to 
    # compute clear sky line
    good_histo_values = np.where((blue<rmax)&(blue>rmin0), blue, 0)
    if np.count_nonzero(good_histo_values) > 500:
        rmin = np.min(good_histo_values[good_histo_values!=0]) # starts binning where we have good data
        # computes the histogram for a single blue image
        (means, edges, numbers)=stats.binned_statistic(blue.flatten(), 
                blue.flatten(), statistic='mean', 
                bins=50, range=(int(rmin),int(rmax)))
        
        histo_labels_reshaped = np.reshape(numbers, (blue.shape[0],blue.shape[1]))
        return histo_labels_reshaped
    
    else:
        # we return None here to signal that we need to use the 
        # mean slope and intercept for the good clear skylines
        return np.ones(blue.shape)*np.nan
    
def get_bin_means(img, histo_labels_reshaped, n=20):
    """
    Takes the same img as get_histo_labels and the histogram index array. 
    Only computes means for bins with at least n values and only takes the
    highest n values in each bin to compute the mean. n is hardcoded to 20
    in Zhu code.
    
    Args:
        img (numpy array): the input image, part of a time series of images
        histo_labels_reshaped: array of same shape as the img bands
        
    Returns:
        a tuple of two lists, the blue means and the read means
    """
    blue = img[:,:,0]
    red = img[:,:,2]

    red_means=[]
    blue_means=[]
    # removing last element because for some reason there was an extra bin in the python version compared to idl
    for i in np.unique(histo_labels_reshaped)[0:-1]:

        red_vals = red[histo_labels_reshaped==i]
        blue_vals = blue[histo_labels_reshaped==i]
        # Zhu set this thresh for number of values needed in bin to compute mean
        if len(blue_vals) >= n: 
            # before selecting top 20, reject outliers based on 
            # red values and pair with corresponding blue values as per Zhu code
            (red_vals_no_outliers, blue_vals_no_outliers) = reject_outliers_by_mean(red_vals, blue_vals)

            ## added these steps from Zhu code, but not sure if/why they are necessary
            # they result in fewer values being averaged in each bin sometimes
            # need to sort by red and use same sorting for blue to keep pairs together
            sort_indices = np.argsort(red_vals_no_outliers)
            red_vals_sorted = red_vals_no_outliers[sort_indices]
            blue_vals_sorted = blue_vals_no_outliers[sort_indices]
            select_n = min([n, ceil(.01*len(blue_vals))])
            red_selected = red_vals_sorted[-select_n:]
            blue_selected = blue_vals_sorted[-select_n:]
            ##
            #finds the highest red values and takes mean
            red_means.append(
                np.mean(
                    red_selected
                )
            )
            blue_means.append(
                np.mean(
                    blue_selected
                )
            )
    return (blue_means, red_means)
    
    
def get_intercept_and_slope(blue_means, red_means, histo_labels_reshaped, nbins):
    """
    Takes the mean lists, the histogram labels, and nbins and computes the intercept
    and slope. includes logic for dealing with too few bins and if the slope that
    is computed is too low.
    
    Args:
        blue_means (list): means of the bins for the blue band
        red_means (list): means of the bins for the red band
        histo_labels_reshaped: array of same shape as the img bands
        
    Returns:
        a tuple of two floats, the intercept and the slope.
    """
    # we want at least half of our ideal data points to construct the clear sky line
    if len(np.unique(histo_labels_reshaped)) > .5 * nbins:
        #followed structure of this example: https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html
        model = statsmodels.formula.api.quantreg('reds~blues', {'reds':red_means, 'blues':blue_means})

        result = model.fit()

        intercept = result.params[0]
        slope = result.params[1]
        # mark as mean, later this is filled with mean slope and mean intercept
        if slope < 1.5:
            return (np.nan,np.nan)
        return (intercept, slope)
    # if we don't have even half the ideal amount of bin means...
    # mark as mean, later this is filled with mean slope and mean intercept
    else: 
        return (np.nan, np.nan)
    
def get_clear_skyline(img, rmin0, rmax, nbins=50):
    """
    Computes the clear sky line for a single image using the
    automatic bin based approach used by Zhen and Elmer 2018.
    Returns the slope and intercept of the clear sky line.
    Larger images are easier to compute a clear sky line, 
    smaller images with more clouds are more difficult and may
    need to take an assumed slope or both slope and intercept.
    This function puts the steps together. 
    
    Args:
        img (numpy array): bgrnir array
        rmin0 (int): minimum edge of the histogram (later adjusted based on each image)
        rmax (int): maximum edge of the histogram
        nbins (int): number of histogram bins
    
    Returns:
        tuple of nan if there are not enough good values to compute 
        a histogram
        
        or
        
        a tuple with the intercept and slope of the clear sky line.
        See get_intercept_and_slope for logic on how intercept and slope
        is computed with different edge cases
    """

    histo_labels_reshaped = get_histo_labels(img, rmin0, rmax, nbins)
    if np.isnan(histo_labels_reshaped).all() == True:
        return (np.nan, np.nan)
    
    blue_means, red_means = get_bin_means(img, histo_labels_reshaped)
    
    intercept, slope = get_intercept_and_slope(blue_means, red_means, histo_labels_reshaped, nbins)
    
    return (intercept, slope)
    
def compute_hot_series(t_series, rmin, rmax, n_bin=50):
    """Haze Optimized Transformation (HOT) test
    Equation 3 (Zhu and Woodcock, 2012)
    Based on the premise that the visible bands for most land surfaces
    are highly correlated, but the spectral response to haze and thin cloud
    is different between the blue and red wavelengths.
    Zhang et al. (2002)
    In this implementation, the slope (a) and intercept(b)
    of the clear sky line are computed automatically using a bin based approach.

    Parameters
    ----------
    t_series: a 4D array with the band index as the third axis, image index as
    the fourth axis (counting from 1st).

    Output
    ------
    ndarray: The values of the HOT index for the image, a 3D array
    """
    blues = t_series[:,:,0,:]
    reds = t_series[:,:, 2,:]
    intercepts_slopes = np.array(
        list(map(lambda x: get_clear_skyline(x,rmin,rmax),
                np.moveaxis(t_series,3,0)))
        )
    # assigns slope and intercept if an image is too cloudy (doesn't have 500 pixels in rmin, rmax range)
    if np.isnan(intercepts_slopes).all():
        # extreme case where no images can get a clear sky line
        intercepts_slopes[:,1] = 1.5
        intercepts_slopes[:,0] = 0
    if np.isnan(intercepts_slopes).any():
        # case where some images can't get a clear skyline
        intercepts_slopes[:,1][np.isnan(intercepts_slopes[:,1])] = np.nanmean(intercepts_slopes[:,1])
        intercepts_slopes[:,0][np.isnan(intercepts_slopes[:,0])] = np.nanmean(intercepts_slopes[:,0])
    def helper(blue, red, ba):
        b,a = ba
        return abs(blue*a - red+b)/np.sqrt(1.0+a**2)
    # map uses the first axis as the axis to step along
    # need to use lambda to use multiple args
    hot_t_series = np.array(list(map(lambda x,y,z: helper(x,y,z), 
                    np.moveaxis(blues,2,0), 
                    np.moveaxis(reds,2,0), 
                    intercepts_slopes)))
    return hot_t_series, intercepts_slopes

def reassign_labels(class_img, cluster_centers, k=3):
    """Reassigns mask labels of t series
    based on magnitude of the cluster centers.
    This assumes land will always be less than thin
    cloud which will always be less than thick cloud,
    in HOT units"""
    idx = np.argsort(cluster_centers.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(k)
    return lut[class_img]

def sample_and_kmeans(hot_t_series, hard_hot=6000, sample_size=10000):
    """Trains a kmeans model on a sample of the time series
    and runs prediction on the time series.
    A hard coded threshold for the hot index, hard_hot, is
    for allowing the kmeans model to capture more variation 
    throughout the time series. Without it, kmeans is skewed toward
    extremely high HOT values and classifies most of the time series
    as not cloudy."""
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # kmeans centers differ slightly due to the method of initialization.
    # Zhu used mean and standard deviation fo the systematic sample, we use kmeans++
    km = KMeans(n_clusters=3, n_init=1, max_iter=50, tol=1e-4, n_jobs=-1, 
                      verbose=False, random_state=4)

#     sample_values = np.random.choice(
#         hot_t_series.flatten()[hot_t_series.flatten()<hard_hot], 
#         size=sample_size).reshape(-1,1)
    interval = int(len(hot_t_series.flatten())/sample_size)
    sample_values = hot_t_series.flatten()[::interval].reshape(-1,1)
    
    fit_result = km.fit(sample_values)
    
    predicted_series = fit_result.predict(hot_t_series.flatten().reshape(-1,1)).reshape(hot_t_series.shape)
    
    return reassign_labels(predicted_series, fit_result.cluster_centers_, k=3), fit_result.cluster_centers_

def calculate_upper_thresh(hot_t_series, cloud_masks, A_cloud):
    """Uses temporal refinement as defined by Zhu and Elmer 2018
    to catch thin clouds by defining the upper boundary, U for clear 
    pixels. Later we might want to compute a neighborhood std 
    through the t_series."""
    hot_potential_clear = np.array(list(map(
        lambda x, y: np.where(x>0, np.nan, y),
        cloud_masks, hot_t_series))) # set cloud to nan
    hot_potential_cloudy = np.array(list(map(
        lambda x, y: np.where(x==0, np.nan, y),
        cloud_masks, hot_t_series))) # set non cloud to nan
    t_series_std = np.nanstd(hot_potential_clear, axis=0)
    t_series_mean = np.nanmean(hot_potential_clear, axis=0)
    t_series_min = np.nanmin(hot_potential_clear, axis=0)
    t_series_max = np.nanmax(hot_potential_clear, axis=0)
    range_arr = t_series_max - t_series_min
    
    # cloud_series_min can be computed more efficiently using k means centers 
    # if a single k means model is used
    # according to Zhu in personal communciation. This is done in IDL code
    
    # NRDI (adjust_T in the IDL code) is a problem here because the HOT indices 
    # vary a lot in the planet images. if we train a kmeans model for each image
    # Th_initial will have a very low initial value, if we train one kmeans model
    # then the model will produce innacurate initial masks because of extremely high
    # HOT values. Need to find a work around.
    
    # the sticky point is how cloud_series_min is calculated. if it is the minimum
    # of all cloudy areas calculated by multiple kmeans models, it is not correct for 
    # the whole t series
    
    cloud_series_min = np.nanmin(hot_potential_cloudy.flatten(), axis=0)
    
    NRDI = (cloud_series_min - range_arr)/(cloud_series_min + range_arr)
    upper_thresh_arr = t_series_mean + (A_cloud+NRDI)*t_series_std
    
    return (upper_thresh_arr, hot_potential_clear, hot_potential_cloudy)

def apply_upper_thresh(t_series, hot_t_series, upper_thresh_arr, initial_kmeans_clouds, hot_potential_clear, hot_potential_cloudy, dn_max):
    """Applies the masking logic to refine the initial cloud
    masks from k-means using the global threshold and 
    upper threshold computed from the time series.
    Returns a time series of refined masks where 2 is cloud and 1 is clear land."""
    
    cloud_series_mean_global = np.nanmean(hot_potential_cloudy.flatten(), axis=0)
    cloud_series_std_global = np.nanstd(hot_potential_cloudy.flatten(), axis=0)
    global_cloud_thresh = cloud_series_mean_global - 1.0*cloud_series_std_global
    # 0 is where hot is below upper threshold, 1 is above
    #testing fix, misse dthat I need to use this thresh here to refine
    #initial clouds by setting pixels to not cloudy if the HOT values are too low
    #refine initial cloud
    initial_kmeans_clouds_binary = np.where(initial_kmeans_clouds > 0, 2 , 1)
    refined_masks = np.where(np.less(hot_potential_cloudy, upper_thresh_arr), 1, initial_kmeans_clouds_binary)
    # add missed clouds
    refined_masks = np.where(np.logical_and(np.greater(hot_potential_clear, upper_thresh_arr), reshape_as_raster(np.greater(t_series[:,:,3,:],dn_max*.1))), 2, refined_masks)
    
    # global_thresh_arr = np.ones(refined_masks.shape)*global_cloud_thresh doesn't have much impact in intiial tests
    
    refined_masks = np.where(hot_t_series > global_cloud_thresh, 2, refined_masks)
    
    return refined_masks

def cloud_height_min_max(angles, longest_d, shortest_d):
    """Calculates the range of possible cloud heights using the 
    scene metadata. The longest distance between a shadow and cloud
    specified in the config cannot be larger than the number of rows 
    or columns in the image.
    
    Args:
        angles (numpy array): 1st column is sun elevation, 2nd is azimuth 
    """
    angles = angles/180.0*3.1415926
    h_high=longest_d/(((np.tan(angles[:,0])*np.sin(angles[:,1]))**2+(np.tan(angles[:,0])*np.cos(angles[:,1]))**2)**0.5)
    h_low=shortest_d/(((np.tan(angles[:,0])*np.sin(angles[:,1]))**2+(np.tan(angles[:,0])*np.cos(angles[:,1]))**2)**0.5)
    return h_high, h_low

def cloud_height_ranges(h_high, h_low):
    """
    Takes two arrays of the max cloud height and minimum cloud height, 
    returning a list of arrays the same length as the time series 
    containing the range of cloud heights used to compute the cloud shadow masks.
    
    Returns: Difference between heighest potential height and lowest, in pixel units.
    """
    h_range_lengths = np.ceil((h_high-h_low)/3.0)
    h_ranges = []
    for i,x in enumerate(h_range_lengths):
        h_ranges.append(np.arange(x)*3+h_low[i])
    return h_ranges

def shadow_shift_coords(h_ranges, angles):
    """
    Computes the possible minimum and maximum x and y magnitudes and 
    directions (in a cartesian sense) for shadows for each scene based 
    on the scene geometry with the sun. Used to determine the direction of the shadow.
    
    Args:
        h_ranges (list of numpy arrays): the ranges of cloud heights for 
            each scene, same length as time series
        angles (numpy array): the sun elevation and azimuth angles. 
            column 0 is sun elevation, 1 is azimuth
    Returns:
        The ending x and y direction and magnitude of the 
            potential shadow relative to the cloud mask
    """
    angles = angles/180.0*3.1415926
    end_x1s = []
    end_y1s = []
    for i, heights in enumerate(h_ranges):      
        end_x1s.append(int(round(-heights[-1]*np.tan(angles[i,0])*np.sin(angles[i,1]))))
        end_y1s.append(int(round(heights[-1]*np.tan(angles[i,0])*np.cos(angles[i,1]))))
    return list(zip(end_x1s, end_y1s))

def make_rectangular_struct(shift_coord_pair):
    """
    Makes the rectangular array with the line structure for dilation int he cloud shadow direction.
    Expects the ending x and y coordinate in array index format for the maximal cloud shadow at the
    maximal cloud height. Array index format means positive y indicates the shadow is south of the cloud, 
    positive x means the shadow is more east of the cloud. rr and cc are are intermediate arrays that store 
    the indices of the line. This line will run from the center of the struct to a corner of the array that 
    is opposite from the direction of the dilation.
    
    Args:
        shift_coord_pair (tuple): Contains the following
            shift_x (int): The maximum amount of pixels to shift the cloud mask in the x direction
            shift_y (int): The maximum amount of pixels to shift the cloud mask in the y direction
    Returns: The struct used by the skimage.morphology.dilation to get the potential shadow mask for a single
                image.
        
    """
    shift_x, shift_y = shift_coord_pair
    struct = np.zeros((abs(shift_y)*2+1, abs(shift_x)*2+1))
    
    if shift_x < 0 and shift_y < 0:
        rr, cc = line(int(abs(shift_y)),int(abs(shift_x)), abs(shift_y)*2, abs(shift_x)*2)  
    elif shift_x < 0 and shift_y > 0:
        rr, cc = line(int(abs(shift_y)),int(abs(shift_x)), 0, abs(shift_x)*2)
    elif shift_x > 0 and shift_y > 0:
        rr, cc = line(int(abs(shift_y)),int(abs(shift_x)), 0, 0)   
    elif shift_x > 0 and shift_y < 0:
        rr, cc = line(int(abs(shift_y)),int(abs(shift_x)), abs(shift_y)*2, 0)   
    struct[rr,cc] = 1
    # removes columns and rows with only zeros, doesn't seem to have an affect
    # struct = struct[~np.all(struct == 0, axis=1)]
    # struct = struct[:, ~np.all(struct == 0, axis=0)]
    return struct

def potential_shadow(struct, cloud_mask):
    """
    Makes the shadow mask from the struct and the cloud mask
    """
    d = dilation(cloud_mask==2, selem=struct)
    d = np.where(d==1, 0, 1)
    d = np.where(cloud_mask==2, 2, d)
    return d

def make_potential_shadow_masks(shift_coords, cloud_masks):
    structs = []
    
    for i in shift_coords:
        structs.append(make_rectangular_struct(i))
        
    shadow_masks = list(map(lambda x, y: potential_shadow(x,y), structs, cloud_masks))
    return np.stack(shadow_masks, axis=0), structs

def make_potential_shadow_masks_multi(shift_coords, cloud_masks):
    args_list = []
    
    for i,coord in enumerate(shift_coords):
        args_list.append((make_rectangular_struct(coord),cloud_masks[i]))
        
    shadow_masks = list(map_processes(potential_shadow, args_list))
    return np.stack(shadow_masks, axis=0)


def apply_li_threshold_multi(shadow_inds, potential_shadow_masks):
    args_list = list(zip(shadow_inds,potential_shadow_masks))
    refined_shadow_masks = list(map_processes(apply_li_threshold, args_list))
    return np.stack(refined_shadow_masks, axis=0)

def min_cloud_nir(masks, t_series):
    """
    Gets the nir band of the scene with the minimum amount of cloud. 
    This will need to be reworked to handle partial scenes so that
    only full scenes are used.
    
    Args:
        masks (numpy array): a 3D array of masks of shape 
                        (count, height, width)
        t_series (numpy array): array of shape (height, width, bands, count) ordered RGBNIR
    Returns (tuple): the nir band of the scene with the least clouds and the index of this scene in the t series
    """
    assert np.unique(masks[0])[-1] == 2
    cloud_counts = [(i==2).sum() for i in masks]
    min_index = np.argmin(cloud_counts)
    return t_series[:,:,3,min_index], min_index  # 3 is NIR

def gain_and_bias(potential_shadow_masks, nir, clearest_land_nir, clearest_index, nir_index):
    """
    Calculates gain for a single imag ein the time series relative to the clearest land image
        Args:
            potential_shadow_masks (numpy array): masks of shape (count, height, width) where 0 is shadow, 1 is clear, 2 is cloud
            nir (numpy array): nir band of the scene to compute gain and bias for
            clearest_land_nir: nir band of the clearest scene
            clearest_index: index for clearest nir band, used for filtering with masks
            nir_index: index for the other nir band
        Returns (tuple): (gain, bias)
    """
    
    # index 3 is NIR, 1 in the mask is clear land
    both_clear = (potential_shadow_masks[clearest_index]==1) & (potential_shadow_masks[nir_index]==1)
    if both_clear.sum() > 100:
        clearest = clearest_land_nir[both_clear]
        nir = nir[both_clear]
        gain = np.std(clearest)/np.std(nir)
        bias  = np.mean(clearest) - np.mean(nir) * gain
    else:
        gain = 1
        bias = 0
    return gain, bias

def gains_and_biases(potential_shadow_masks, t_series, clear_land_nir, clear_land_index):
    gains_biases = []
    for i in np.arange(t_series.shape[-1]):
        gain, bias = gain_and_bias(potential_shadow_masks, t_series[:,:,3,i], clear_land_nir, clear_land_index, i)
        gains_biases.append((gain,bias))
    return gains_biases

def shadow_index_land(potential_shadow_masks, t_series, gains_biases):
    """
    Applies gain and bias to get shadow index from nir for each scene in t series.
    
    Returns (numpy array): shape (count, height, width) of the nir band shadow index 
                where there was previously calculated to be potential shadow
    """
    shadow_inds = []
    for i in np.arange(t_series.shape[-1]):
        # applies calcualtion only where mask says there is not cloud 
        # might need to do this differently for water
        shadow_inds.append(np.where(potential_shadow_masks[i]!=2, t_series[:,:,3,i]*gains_biases[i][0]+gains_biases[i][1], np.nan))
        
    return np.stack(shadow_inds)

def apply_li_threshold(shadow_index, potential_shadow_mask):
    """
    Applies a Li threshold to the cloud masked shadow index
    and subsets this binarized thresholded array to the first 
    potential shadow mask, refining potential shadow regions.
    
    skimage.filters.try_all_threshold showed that Li's threshold was far superior
    to Otsu and other methods. This replaces IDL's use of Inverse Distance Weighting 
    to refine the shadow mask before kmeans clustering since it is faster and returns better results.
    
    Args:
        shadow_index (numpy array): output from shadow_index_land for a single scene that has clouds set to NaN
        potential_shadow_mask (numpy array): the shadow and cloud mask, used to refine the thresholded mask
    
    https://www.sciencedirect.com/science/article/pii/003132039390115D?via%3Dihub
    """

    thresh = threshold_li(shadow_index)

    binary = shadow_index > thresh

    binary = np.where(potential_shadow_mask==0, binary, 1)

    return opening(binary)

if __name__== "__main__":

    import time
    start = time.time()

    ###porting code from original idl written by Xiaolin Zhu
    path_id = "savanna"
    img_path = "/home/rave/cloud-free-planet/cfg/buffered_stacked/"+ path_id+"_stacked.tif"
    angles_path = os.path.join("/home/rave/cloud-free-planet/cfg/buffered_angles", path_id+'_angles_larger_utm.txt')
    result_path = "/home/rave/cloud-free-planet/cfg/atsa_results/" +path_id+"_cloud_and_shadow_masks.tif"
    configs = pyatsa_configs.ATSA_Configs(img_path, angles_path)

    angles = np.genfromtxt(angles_path, delimiter=' ')

    hot_t_series, intercepts_slopes = compute_hot_series(configs.t_series, configs.rmin, configs.rmax)

    initial_kmeans_clouds, kmeans_centers = sample_and_kmeans(hot_t_series, hard_hot=5000, sample_size=10000)

    upper_thresh_arr, hot_potential_clear, hot_potential_cloudy = calculate_upper_thresh(hot_t_series, initial_kmeans_clouds, configs.A_cloud)

    refined_masks = apply_upper_thresh(configs.t_series, hot_t_series, upper_thresh_arr, initial_kmeans_clouds, hot_potential_clear, hot_potential_cloudy, configs.dn_max)

    # axis 0 must be the image count axis, not height or width
    # refined_masks = np.apply_along_axis(opening, 0, refined_masks) # removes single pixel clouds

    # refined_masks = np.apply_along_axis(lambda x: dilation(x, selem=np.ones(5,5)), 0, refined_masks)

    for i in np.arange(refined_masks.shape[0]):
        refined_masks[i] = opening(refined_masks[i], np.ones((5,5)))
    
        # before dilating we need to check for water. where statement currnetly contains hardcoded value to deal with intermittent water being 
        # misclassified as cloud due o HOT index not working over water. We can't generate an accurate water mask with Planet alone because
        # it does not have shortwave infrared. see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018RG000598 
        refined_masks[i] = np.where((configs.t_series[:,:,3,i]<2000)&(refined_masks[i]==2), 1, refined_masks[i])
    
        refined_masks[i] = dilation(refined_masks[i], np.ones((5,5)))

    print("seconds ", time.time()-start)
    print("finished cloud masking")

    start = time.time()

    h_high, h_low = cloud_height_min_max(angles, configs.longest_d, configs.shortest_d)
    h_ranges = cloud_height_ranges(h_high, h_low)
    shift_coords = shadow_shift_coords(h_ranges, angles)
    potential_shadow_masks = make_potential_shadow_masks_multi(shift_coords, refined_masks)

    print("seconds ", time.time()-start)
    print("finished potential shadow masking")

    start = time.time()

    clearest_land_nir, clearest_land_index = min_cloud_nir(potential_shadow_masks, configs.t_series)
    gains_biases = gains_and_biases(potential_shadow_masks, configs.t_series, clearest_land_nir, clearest_land_index)
    shadow_inds = shadow_index_land(potential_shadow_masks, configs.t_series, gains_biases)
    li_refined_shadow_masks = apply_li_threshold_multi(shadow_inds, potential_shadow_masks)

    cloud_shadow_masks = np.where(li_refined_shadow_masks==0, 0, refined_masks) #2 is cloud, 1 is clear land, 0 is shadow

    skio.imsave(result_path,cloud_shadow_masks)

    print("seconds ", time.time()-start)
    print("finished refined shadow masking")