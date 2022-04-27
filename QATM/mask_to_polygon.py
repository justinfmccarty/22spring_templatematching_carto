from shapely.geometry import Polygon #import shapely
import cv2 # import opencv

# source 
# this is a very simple method
# the resulting shape might not be oriented correctly 
# if this is the case then play around with inverting the contour
# https://stackoverflow.com/questions/57965493/how-to-convert-numpy-arrays-obtained-from-cv2-findcontours-to-shapely-polygons

def mask_to_polygon(img):
    
    # apply the cv2 find contours function to the "img" (this is the binary mask array you created)
    contours = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

    # cv2 returns the contour file with an extra dimension, so you just need to simplfy that array
    contour = np.squeeze(contours[0])

    # then just apply the shapely Polygon function to the returned contours.
    polygon = Polygon(contour)

    return polygon
    # ffrom there you should be able to save the polygon to a .shp file. 