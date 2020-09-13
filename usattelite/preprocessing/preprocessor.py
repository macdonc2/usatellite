import numpy as np
import cv2


class Preprocess:

    """
    This is a class to define and apply a top hat filter using cv2.  The user specifies a filter size when instantiating
    a TopHatFilter object.  The object then has the apply method, which takes a dsm_array to be filtered.
    """

    def top_hat_filter(self, dsm_array, filter_size):

        """
        This method applies the top hat filter to the dsm_array.
        """

        filt_size = (filter_size, filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filt_size) 
            
        top_hat_array = np.zeros_like(dsm_array)
            
        for i in range(len(dsm_array)):
            top_hat_array[i] = cv2.morphologyEx(dsm_array[i], cv2.MORPH_TOPHAT, kernel) 
            
        return top_hat_array

    def ndvi(self, data):

        """
        The normalized difference vegetation index is a common measurement used in remote sensing.  Need 
        to check this more carefully...  Michael is differencing multidimensional data?
        """

        return (data[:,:,:,0]-data[:,:,:,1])/(data[:,:,:,0]+data[:,:,:,1])

    def unit_normalize_dims(self, data):
        
        normalized_data = data

        for i in range(np.shape(data)):
            normalized_data -= np.mean(normalized_data[:,:,:,i])
            normalized_data /= np.std(normalized_data)
        
        return normalized_data