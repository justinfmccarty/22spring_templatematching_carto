import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import env_checker
import seaborn as sns
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
import random
import time
from rasterio.features import rasterize
from IPython import display
import geopandas as gpd
import pandas as pd
from skimage.feature import match_template

def crop_image(image,y_start,x_start,height,width):
    return image[y_start:y_start+height,x_start:x_start+width]

def rect_out_rast(y,x,height,width):
    p = Polygon([[x, y], [x+width, y+0], [x+width, y+height], [x+0, y+height]])
    return rasterize([p],out_shape=(width,height)).T

def Sum_of_squared_differences(roi,temp):
    ssd = np.sum((np.array(roi, dtype=np.float32) - np.array(temp, dtype=np.float32))**2)
    return ssd

def Normalized_cross_correlation(roi,temp):
    cc = np.sum(roi*temp)
    nor = np.sqrt(np.sum(roi**2)) * np.sqrt(np.sum(temp**2))
    return cc/nor

def ssd(A,B):
  dif = A.ravel() - B.ravel()
  return np.dot( dif, dif )

def action_rules(action):
    if action==None:
        return action
    elif action<4:
        return action+4
    else:
        return action-4

def tuple_addition(a,b):
    return tuple(map(sum, zip(a, b)))

def get_action_meanings():
        return {0: "South", 1: "Southwest", 2: "West", 3: "Northwest",
                4: "North", 5: "Northeast", 6: "East", 7: "Southeast",
                -1: "No Moves"}
        
def calc_termination_point(point,multipliers,dimensions):
            termination_y = point[0] + multipliers[0] * dimensions[0]
            termination_x = point[1] + multipliers[1] * dimensions[1]
            return (termination_y,termination_x)
        

def calc_increment_space(image_size,template_size):
    actual_increments = image_size/template_size
    possible_increments = int(np.floor(actual_increments))
    return np.arange(0,template_size*(possible_increments+1),template_size)

def remove_black(image):
    blue_band = image[:,:,0]
    green_band = image[:,:,1]
    red_band = image[:,:,2]
    blue_mask = blue_band<=80
    green_mask = green_band<=80
    red_mask = red_band<=80
    mask = blue_mask&green_mask&red_mask
    blue_band[mask] = 255
    green_band[mask] = 255
    red_band[mask] = 255
    newimagebgr = np.stack((blue_band,green_band, red_band),axis = 2)
    newimage = np.stack((red_band,green_band, blue_band),axis = 2)
    plt.imshow(newimage)
    cv2.imwrite('newimage.png',newimage)
    return newimagebgr

def remove_white(image):
    blue_band = image[:,:,0]
    green_band = image[:,:,1]
    red_band = image[:,:,2]
    blue_mask = blue_band>=80
    green_mask = green_band>=80
    red_mask = red_band>=80
    mask = blue_mask&green_mask&red_mask
    blue_band[mask] = 255
    green_band[mask] = 255
    red_band[mask] = 255
    newimagebgr = np.stack((blue_band,green_band, red_band),axis = 2)
    newimage = np.stack((red_band,green_band, blue_band),axis = 2)
    plt.imshow(newimage)
    cv2.imwrite('newimage.png',newimage)
    return newimagebgr

def plot_line_string(point_list,ax,reduce_factor=1):
    # have to reverse the order of the points for geopandas as well
    reduced_list = [(int(x[1]*reduce_factor),int(x[0]*reduce_factor)) for x in point_list]
    gpd.GeoSeries(LineString(reduced_list)).plot(ax=ax)
    gpd.GeoSeries(Point(reduced_list[0])).plot(ax=ax)
    
def read_image_to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def remove_white_hsv_meth(img):
    # Convert to HSV colourspace and extract just the Saturation
    Sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    # Find best (Otsu) threshold to divide black from white, and apply it
    _ , mask = cv2.threshold(Sat,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # At each pixel, choose foreground where mask is set and background elsewhere
    return np.where(mask[...,np.newaxis], img, 80)

def resize_image(image, reduce_factor=0.1):
    new_res = (int(image.shape[1]*reduce_factor), int(image.shape[0]*reduce_factor))
    return cv2.resize(image, new_res)
   