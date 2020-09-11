import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
from keras import utils

class Preprocessor:

    """
    The Preprocessor class is available to help prepare data for machine learning.  
    
    
    top_hat_filter will apply a white-top-hat-filter for removing terrain effect.  
    This filter is somewhat sensitive to the kernel size (as is every filter, I suppose),
    and will need some trial and error.  It is assumed that the images are reshaped prior
    to filtering.
    
    categorical leverages Keras to create labels
    """
    
    def top_hat_filter(dsm_array, filter_size=400):
        
        filt_size = (filter_size, filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filt_size) 
        
        top_hat_array = np.zeros_like(dsm_array)
        
        for i in range(len(dsm_array)):
            top_hat_array[i] = cv2.morphologyEx(dsm_array[i], cv2.MORPH_TOPHAT, kernel) 
        
        return top_hat_array
    
    def categorical(labels):
        
        return utils.to_categorical(labels)