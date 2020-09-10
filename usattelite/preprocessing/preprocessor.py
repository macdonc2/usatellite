import numpy as np
import cv2


class TopHatFilter:

    """
    This is a class to define and apply a top hat filter using cv2.  The user specifies a filter size when instantiating
    a TopHatFilter object.  The object then has the apply method, which takes a dsm_array to be filtered.
    """

    def __init__(self, filter_size):
        self.filter_size = filter_size

    def apply(self, dsm_array):
        filt_size = (self.filter_size, self.filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filt_size) 
            
        top_hat_array = np.zeros_like(dsm_array)
            
        for i in range(len(dsm_array)):
            top_hat_array[i] = cv2.morphologyEx(dsm_array[i], cv2.MORPH_TOPHAT, kernel) 
            
        return top_hat_array