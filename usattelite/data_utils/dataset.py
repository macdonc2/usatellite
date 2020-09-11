# imports

import cv2
import tifffile # for reading tiff files. There are other modules that can do this, but tifffile is most stable on Windows
import numpy as np # for array handling
import glob # to gather up image filepath lists
from skimage.transform import resize # we're gonna do some rearranging
import scipy # same
from sklearn.preprocessing import OneHotEncoder
from keras import utils


class DataLoader:
    
    """
    A Dataset class that has methods for loading tiff files of satellite images for semantic segmentation.  
    """

    def load_list(self, directory, pattern):
        """
        Simply loads data given a directory and file pattern.
        """
        
        return glob.glob(f'{directory}/' + f'{pattern}' + '*.tif')

    def load_and_resize_images(self, im_list, shape, dims):
        """
        Resize images and place inside a library for ML tasks.  im_list depends on using load_list prior 
        to executing the method.
        """
        
        shape = shape
        img_array = np.zeros((len(im_list), shape, shape, dims))
        
        
        for i in range(len(im_list)):
            img = tifffile.imread(im_list[i])
            img_array[i] = resize(img, (shape, shape, dims))
            
        return img_array
    
    def load_and_resize_dsm(self, im_list, shape, dims=1):
        """
        Resize images and place inside a library for ML tasks.  im_list depends on using load_list prior 
        to executing the method.
        """

        shape = shape
        dsm_array = np.zeros((len(im_list), shape, shape, dims))


        for i in range(len(im_list)):
            img = tifffile.imread(im_list[i])
            dsm_array[i] = resize(img, (shape, shape, dims))

        return np.squeeze(dsm_array,-1)
    
    def load_and_resize_rgb_labels(self, rgb_list, shape, dims):
        """
        Resize rgb images of labels and place inside a library for ML tasks.  rgb_list depends on using load_list prior 
        to executing the method.  For now the number of classes is fixed at 6 for this dataset.  Figure out a better method
        later.
        """
        
        shape = shape
        rgb_array = np.zeros((len(rgb_list), shape, shape, dims))
        
        
        for i in range(len(rgb_list)):
            img = tifffile.imread(rgb_list[i])
            rgb_array[i] = resize(img, (shape, shape, dims))

        
        rgb_array[rgb_array>=0.5] = 1
        rgb_array[rgb_array<0.5] = 0    
        
        onehot_label_array = np.zeros((len(rgb_list),shape,shape,6), dtype=np.uint8)
        
        for k in range(len(rgb_list)):
            for i in range(shape):
                for j in range(shape):
                    
                    if(rgb_array[k,i,j,0]==1 and rgb_array[k,i,j,1]==1 and rgb_array[k,i,j,2]==1):
                        onehot_label_array[k,i,j,0]=1
                    
                    elif(rgb_array[k,i,j,0]==0 and rgb_array[k,i,j,1]==0 and rgb_array[k,i,j,2]==1):
                        onehot_label_array[k,i,j,1]=1
                    
                    elif(rgb_array[k,i,j,0]==0 and rgb_array[k,i,j,1]==1 and rgb_array[k,i,j,2]==1):
                        onehot_label_array[k,i,j,2]=1
                    
                    elif(rgb_array[k,i,j,0]==0 and rgb_array[k,i,j,1]==1 and rgb_array[k,i,j,2]==0):
                        onehot_label_array[k,i,j,3]=1
                    
                    elif(rgb_array[k,i,j,0]==1 and rgb_array[k,i,j,1]==1 and rgb_array[k,i,j,2]==0):
                        onehot_label_array[k,i,j,4]=1
                    
                    elif(rgb_array[k,i,j,0]==1 and rgb_array[k,i,j,1]==0 and rgb_array[k,i,j,2]==0):
                        onehot_label_array[k,i,j,5]=1


        return onehot_label_array
        
                        