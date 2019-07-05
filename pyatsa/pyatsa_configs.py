import numpy as np
import skimage.io as skio
import os

class ATSA_Configs():
    def __init__(self, image_path, angles_path):
        self.t_series=skio.imread(image_path)
        self.angles = np.genfromtxt(angles_path, delimiter=' ')
        #set the following parameters
        self.dn_max=10000  #maximum value of DN, e.g. 7-bit data is 127, 8-bit is 255
        self.background=0  #DN value of background or missing values, such as SLC-off gaps
        self.buffer=1    #width of buffer applied to detected cloud and shadow, recommend 1 or 2 

        #parameters for HOT caculation and cloud detection
        #------------------------------
        self.n_band=4     # number of bands of each image
        self.n_image=self.t_series.shape[2]/self.n_band   # number of images in the time-series
        self.blue_b=0    # band index of blue band, note: MSS does not have blue, use green as blue
        self.green_b=1   # band index of green band
        self.red_b=2     # band index of red band
        self.nir_b=3     # band index of nir band

        self.A_cloud=2.0 # threshold to identify cloud (mean+A_cloud*sd), recommend 0.5-1.5 for Landsat, smaller values can detect thinner clouds
        self.maxblue_clearland=self.dn_max*0.15 # estimated maximum blue band value for clear land surface
        self.maxnir_clearwater=self.dn_max*0.05 # estimated maximum nir band value for clear water surface
        self.rmax = self.maxblue_clearland # max value for blue band for computing clear line
        self.rmin = .01*self.dn_max # min DN value for blue band for computing clear line
        self.n_bin = 50 # number of bins between rmin and rmax

        #parameters for shadow detection
        #------------------------------
        self.shortest_d=7.0       #shortest distance between shadow and cloud, unit is pixel resolution
        self.longest_d=300.0  #longest distance between shadow and its corresponding cloud, unit is "pixel",can be set empirically by inspecting images
        self.B_shadow=1.5   #threshold to identify shadow (mean-B_shadow*sd), recommend 1-3, smaller values can detect lighter shadows
        #------------------------------

        #we reshape our images that were stacked on the band axis into a 4D array
        self.t_series = np.reshape(self.t_series,(self.t_series.shape[0],self.t_series.shape[1],self.n_band,int(self.n_image)), order='F')

