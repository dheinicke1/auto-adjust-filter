"""
File that takes a flattened color image and fiilter parameters
and returns the image masekd by the filter.

Image processor:
1) Pass in a color image. Blur and sharpen image
2) Convert the image to HSV (Hue, Saturation, Value) format
2) Starting with an initial filter range, create a binary mask 
    for the image, and count the number and size of the 
    countours found
3) (Optional) If the number of contours found does not meet a minimum
    critera passed to function, broaden the filter until contour requirements 
    are met.
4) Return the original color image, with the binary mask applied.

"""
import cv2
import numpy as np
import trafaret as t

from image_filter.filter_functions import tune_sat
from image_filter.filter_functions import sharpen_image
from image_filter.filter_functions import count_contours


class ImageFilter:
    '''
    Methods
        __init__(self, image = [], upper_HSV = [], lower_HSV = [])
    
            Initialize the filter. Required inputs are a color image
            and an initial upper and lower range for the filter.
            
        find_shapes(self)
            
            Method to adjust the filter criteria until a masked image 
            can be retuned
        
    '''
    
    @t.guard(image = t.Any(),
             upper_HSV = t.List(t.Int(gte = 0)),
             lower_HSV = t.List(t.Int(gte = 0)))
    def __init__(self, image = [], upper_HSV = [], lower_HSV = []):
        '''
        Initializer.
        
        Parameters:
        ----------
            image : list
                A list representing a color image of 
                shape (image_size, image_size, 3)
                    
            upper_HSV : list
                A list of the upper limits of the initial HSV
                filter
            
            lower_HSV: list
                A list of the lower limits of the initial HSV
                filter
                
        Returns:
        ----------
            None. Initializes and saves the filter object attributes.
        
        '''
        
        # Build Filter
        self.image = image
        self.upper_HSV = np.array(upper_HSV)
        self.lower_HSV = np.array(lower_HSV)
        self.contour_threshold_upper = 10 # Maximum number of contours
        self.contour_threshold_lower = 1  # Minimum number of contours
        self.min_shape_size = 150         # Minimum contour size to be counted
        self.image_size = 299             # Image size
        self.shape_stepsize = 200         # Step size to reduce minimum contour 
                                          #   size, if the filter can't find minimum
        
        # Option to blur and sharpen the image before masking
        self.sharpen = True
        
        # cv2 blurring methods, as a string. Options are:
        # blur, GaussianBlur, medianBlur, bilateralFilter or none
        self.filter_type = 'bilateralFilter'
        
        ### ERROR CHECKING ###
        
        # Verify upper bounds are greater than lower bounds
        
        if (upper_HSV[0] <= lower_HSV[0] or 
            upper_HSV[1] <= lower_HSV[1] or 
            upper_HSV[2] <= lower_HSV[2]):
               raise ValueError('Each value in HSV Upper range muste be ' 
                                'greater than values in Lower HSV')
                                
        #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
        if (upper_HSV[0] > 180 or 
            lower_HSV[0] > 180):
                raise ValueError('Hue must be less than 180')
                
        if (upper_HSV[1] > 255 or 
            lower_HSV[1] > 255):
                raise ValueError('Saturation must be less than 255')
        
        if (upper_HSV[2] > 255 or 
            lower_HSV[2] > 255):
                raise ValueError('Value must be less than 255')

    def find_shapes(self):
        '''
            Sharpens the image (if self.sharpen is true), resizes the image 
            to specified size and applies the tune_sat function to see
            if enough contours of a minimum size can be found by adjusting 
            the filter boundaries (see filter_functions). If not, the 
            minimum contour size is stepped down, and the tune_sat 
            function tries again.
            
            Once enough contours are found, the masked image is 
            converted back to color. 
            
            Method returns the maked image in color, the list of contours 
            found by cv2.findContours, and the upper and lower filter 
            boundaries found.
        '''     
        contour_counter = 0
        image = self.image
        upper_HSV = self.upper_HSV
        lower_HSV = self.lower_HSV
        contour_threshold_upper = self.contour_threshold_upper
        contour_threshold_lower = self.contour_threshold_lower
        min_shape_size = self.min_shape_size
        image_size = self.image_size
        sharpen = self.sharpen
        shape_stepsize = self.shape_stepsize
        
        image = cv2.resize(image, (image_size, image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if sharpen == True:
            image = sharpen_image(image, method = self.filter_type)
        
        init_sat_upper = upper_HSV[1]
        init_sat_lower = lower_HSV[1]
        
        while contour_counter <= contour_threshold_lower or contour_counter >= contour_threshold_upper:
            
            processed_image, contours, upper_HSV, lower_HSV = tune_sat(image, upper_HSV, lower_HSV, min_shape_size, contour_threshold_upper, contour_threshold_lower)
            
            contour_counter = count_contours(contours, min_shape_size)
            
            min_shape_size = min_shape_size - shape_stepsize
    
            if contour_counter >= contour_threshold_lower and contour_counter <= contour_threshold_upper:
                break
            
            upper_HSV[1] = init_sat_upper
            lower_HSV[1] = init_sat_lower
            
            if min_shape_size < 50:
                print('Contours Found are not in range')
                break
            
        image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)
        upper_HSV = np.ndarray.tolist(upper_HSV)
        lower_HSV = np.ndarray.tolist(lower_HSV)
        
        return(image, contours, upper_HSV, lower_HSV)
        

