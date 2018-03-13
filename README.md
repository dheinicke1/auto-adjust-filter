# Auto Adjust Filter

File that takes a flattened color image and fiilter parameters
and returns the image masekd by the filter.

Created to preprocess the [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) challenge on Kaggle. 

The images to classify contained green plants against the background of the seed tray. After converting the images from color to HSV format, a binary mask can be applied to remove background "noise" from the image. The issue is that the plant seedlings color varies, so a fixed binary mask that works well on one image may filter out some or all of the seedling from another image.

For example, the following plants have been masked by a fiaxed filter and the aauto adjust filter.

Initial Filter HSV range: (24, 27, 15) to (48, 255, 255)

![alt text](https://github.com/dheinicke1/auto-adjust-filter/blob/master/files/Filter_example.png}

The adjustable filter changes the common wheat range to (23, 27, 15) to (49, 255, 255)

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
