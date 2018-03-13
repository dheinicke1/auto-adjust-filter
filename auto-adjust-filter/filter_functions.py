"""
Image processor:
1) Pass in a color image. Blur and sharpen image
2) Convert the image to HSV format
2) Starting with an initial filter range, create a binary mask 
    for the image, and count the number and size of the 
    countours found
3) (Optional) If the number of contours found does not meet a minimum
    critera passed to function, broaden the filter until contour requirements 
    are met.
4) Return the original color image, with the binary mask applied.

"""

import cv2
import warnings

def sharpen_image(image, method=''):
    '''
    Applies cv2 Blurring methods to the image.
    See OpenCV documentation.
    
    Parameters:
    ----------
        image : array
            Image size by imagae size by 3 array in HSV
        method : string
            Name of cv2 method to be applied
        
    Returns:
    -------
        image_sharp : array
            Blurred image
    '''
    
    if method == 'blur':
        image_blurred = cv2.blur(image, (5,5))
    if method == 'GaussianBlur':
        image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    if method == 'medianBlur':
        image_blurred = cv2.medianBlur(image,5)
    if method == 'bilateralFilter':
        image_blurred = cv2.bilateralFilter(image, 9, 75,75)
    else:
        image_blurred = image
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def create_mask(image, upper_HSV, lower_HSV):
    '''
    Create a binary mask from an image in HSV format. 
    Pass an image, upper and lower HSV limits, returns the masked image 
    as well as the number of countrous found by OpenCV's findContours 
    method
    
    Parameters:
    ----------
        image : array
            Image size by imagae size by 3 array in HSV format
        upper_HSV: array
            Numpy array of upper limit of the filter [hue, saturation, value]
        lower_HSV: array
            Numpy array of lower limit of the filter [hue, saturation, value]
    
    Returns:
    -------
        masked_image : array
            Image with binary mask applied
        contours : array
            Array of contour arrays found by cv2.findContours
    '''
    
    mask = cv2.inRange(image, lower_HSV, upper_HSV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    (_, contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return masked_image, contours

def count_contours(contours, min_shape_size):
    """
    Pass the masked image from create_mask as well as a minimum size for
    the contours being counted (to prevent 'noise' contours from being 
    counted). 
    
    Returns the number of contours of a minimum size found.
    
    Parameters:
    ----------
        contours : array
            Array of contour arrays from cv2.findContours
            See Open CV documentation
        
        min_shape_size : int
            Minimum size of contour to be counted. Prevents small 'noise"
            contours from being counted when adjustnig the filter
    
    Returns:
    -------
        num : int
            Number of contours foubd
    """
    num = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_shape_size:
            num = num + 1
    return(num)
    
def tune_hue(image, 
             upper_HSV,
             lower_HSV,
             min_shape_size,
             contour_threshold_upper,
             contour_threshold_lower):
    '''
    If the specifications of the contours found do not meet minimum 
    criteria, incrementally widens the hue range of the filter by 2
    
    If the hue range drops below 20, break
    
    Parameters:
    ----------
        image : array
            Image size by imagae size by 3 array in HSV
            upper_HSV: array
            Numpy array of upper limit of the filter [hue, saturation, value]
        
        upper_HSV: array
            Numpy array of lower limit of the filter [hue, saturation, value]
            
        lower_HSV: array
            Numpy array of lower limit of the filter [hue, saturation, value]
        
        min_shape_size : int
            Minimum size of contour to be counted. Prevents small 'noise"
            contours from being counted when adjustnig the filter
        
        contour_threshold_upper : int
            Maximum number of contours to be found to stop widening the filter
        
        contour_threshold_lower : int
            Minimum number of contours to be found to stop widening the filter
        
    Returns:
    -------
        tuned_imgae : array
            Image with binary mask applied
        
        contours : array
            Array of contour arrays found by cv2.findContours
            
        upper_HSV: array
            Upper limit of the filter [hue, saturation, value] after being 
            adjusted by tube_hue
            
        lower_HSV: array
            Lower limit of the filter [hue, saturation, value] after being 
            adjusted by tube_hue
    '''
    
    contour_counter = 0

    while contour_counter <= contour_threshold_lower or contour_counter >= contour_threshold_upper:
        tuned_imgae, contours = create_mask(image, upper_HSV, lower_HSV)
        contour_counter = count_contours(contours, min_shape_size)
        
        lower_HSV[0] = lower_HSV[0] - 1
        upper_HSV[0] = upper_HSV[0] + 1
        
        if lower_HSV[0] < 20:
            break

    return(tuned_imgae, contours, upper_HSV, lower_HSV)
        
def tune_sat(image, 
             upper_HSV, 
             lower_HSV, 
             min_shape_size, 
             contour_threshold_upper, 
             contour_threshold_lower):
    '''
    Initially calls tune_hue.
    
    If the specifications of the contours found do not meet minimum 
    criteria after the minimum hue has reached 20: 
    
    1) Reset the hue to initial values
    
    2) Incrementally narrow the saturation range of the filter by 5 
        and tune hue again
        
    If the saturation range drops below 45, break
    
    Parameters:
    ----------
        image : array
            Image size by imagae size by 3 array in HSV
            upper_HSV: array
            Numpy array of upper limit of the filter [hue, saturation, value]
        
        upper_HSV: array
            Numpy array of lower limit of the filter [hue, saturation, value]
            
        lower_HSV: array
            Numpy array of lower limit of the filter [hue, saturation, value]
        
        min_shape_size : int
            Minimum size of contour to be counted. Prevents small 'noise"
            contours from being counted when adjustnig the filter
        
        contour_threshold_upper : int
            Maximum number of contours to be found to stop widening the filter
        
        contour_threshold_lower : int
            Minimum number of contours to be found to stop widening the filter
        
    Returns:
    -------
        tuned_imgae : array
            Image with binary mask applied
        
        contours : array
            Array of contour arrays found by cv2.findContours
            
        upper_HSV: array
            Upper limit of the filter [hue, saturation, value] after being 
            adjusted by tune_sat
            
        lower_HSV: array
            Lower limit of the filter [hue, saturation, value] after being 
            adjusted by tune_sat
    '''
    
    contour_counter = 0
    init_hue_upper = upper_HSV[0]
    init_hue_lower = lower_HSV[0]
    
    while contour_counter <= contour_threshold_lower or contour_counter >= contour_threshold_upper:
        
        tuned_imgae, contours, upper_HSV, lower_HSV = tune_hue(image, upper_HSV, lower_HSV, min_shape_size, contour_threshold_upper, contour_threshold_lower)
        
        contour_counter = count_contours(contours, min_shape_size)
        
        if contour_counter >= contour_threshold_lower and contour_counter <= contour_threshold_upper:
            break
        
        lower_HSV[1] = lower_HSV[1] - 5
        
        upper_HSV[0] = init_hue_upper
        lower_HSV[0] = init_hue_lower
        
        if lower_HSV[1] <= 45:
            break
    
    return(tuned_imgae, contours, upper_HSV, lower_HSV)    
