import os
from shapely.geometry import Polygon
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import rasterio
import geopandas as gpd
import rioxarray as rxr
import multiprocess as mp
import shapely
from shapely.ops import unary_union
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from seaborn import color_palette
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
from utils import *
# %matplotlib inline


"""Conver image to tensor"""
class ImageDataset_og(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_path = list(template_dir_path.iterdir())
        self.image_name = image_name
        
        self.image_raw = cv2.imread(self.image_name)
        
        self.thresh_df = None
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)
            
        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)
        
    def __len__(self):
        return len(self.template_names)
    
    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template = cv2.imread(template_path)
        if self.transform:
            template = self.transform(template)
        thresh = 0.7
        if self.thresh_df is not None:
            if self.thresh_df.path.isin([template_path]).sum() > 0:
                thresh = float(self.thresh_df[self.thresh_df.path==template_path].thresh)
        return {'image': self.image, 
                    'image_raw': self.image_raw, 
                    'image_name': self.image_name,
                    'template': template.unsqueeze(0), 
                    'template_name': template_path, 
                    'template_h': template.size()[-2],
                   'template_w': template.size()[-1],
                   'thresh': thresh}

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_path = list(template_dir_path.iterdir())
        self.image_name = image_name
        # with rasterio.open(self.image_name) as f:
        #   self.image_raw = f.read()
        image_crop = cv2.imread(self.image_name)
        im_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        #print(im_rgb.shape)
        self.image_raw = im_rgb[5600:7800,8000:10400,:]
        #self.image_raw = remove_black(self.image_raw)
        #self.image_raw = image_crop
        
        self.thresh_df = None
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)
            
        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)
            # print(self.image.shape)
        
    def __len__(self):
        return len(self.template_names)
    
    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template = cv2.imread(template_path)
        #template = cv2.imread("./template/template1_1.png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        # with rasterio.open(template_path) as f:
        #   template = f.read()
        if self.transform:
            template = self.transform(template)
        thresh = 0.7
        if self.thresh_df is not None:
            if self.thresh_df.path.isin([template_path]).sum() > 0:
                thresh = float(self.thresh_df[self.thresh_df.path==template_path].thresh)
        return {'image': self.image, 
                    'image_raw': self.image_raw, 
                    'image_name': self.image_name,
                    'template': template.unsqueeze(0), 
                    'template_name': template_path, 
                    'template_h': template.size()[-2],
                   'template_w': template.size()[-1],
                   'thresh': thresh}

class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model= copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)
        
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        
    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        if mode=='big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        else:        
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)

class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _ , H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]

class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None
    def __call__(self, template, image, image_name):
        T_feat = self.featex(template)
        if self.I_feat_name is not image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True), T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps

class QATM():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        print(batch_size)
        x = x.view(batch_size, ref_row*ref_col, qry_row*qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha*xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row*ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values
    def compute_output_shape( self, input_shape ):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)

def nms(score, w_ini, h_ini, thresh=0.4):
    dots = np.array(np.where(score > thresh*score.max()))
    
    x1 = dots[1] - w_ini//2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini//2
    y2 = y1 + h_ini

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes

def plot_result(image_raw, boxes, show=False, save_name=None, color=(255, 0, 0)):
    # plot result
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 3)
    if show:
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img

"""Multi"""

def nms_multi(scores, w_array, h_array, thresh_list):
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    # omit not-matching templates
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]
    # extract candidate pixels from scores
    dots = None
    dos_indices = None
    for index, score in zip(indices_omit, scores_omit):
        dot = np.array(np.where(score > thresh_list[index]*score.max()))
        if dots is None:
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    dots_indices = dots_indices.astype(np.int)
    x1 = dots[1] - w_array[dots_indices]//2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices]//2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    dots_indices = dots_indices[order]

    keep = []
    keep_index = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
        
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2,0,1)
    return boxes, np.array(keep_index)

def plot_result_multi(image_raw, boxes, indices, show=False, save_name=None, color_list=None):
    d_img = image_raw.copy()
    if color_list is None:
        color_list = color_palette("hls", indices.max()+1)
        color_list = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), color_list))
    for i in range(len(indices)):
        d_img = plot_result(d_img, boxes[i][None, :,:].copy(), color=color_list[indices[i]])
    if show:
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img

"""Running"""

def run_one_sample(model, template, image, image_name):
    val = model(template, image, image_name)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)
    
    batch_size = val.shape[0]
    
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i,:,:,0]
        gray = cv2.resize( gray, (image.size()[-1], image.size()[-2]) )
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score( gray, w, h) 
        score[score>-1e-7] = score.min()
        score = np.exp(score / (h*w)) # reverse number range back after computing geometry average
        scores.append(score)
    return np.array(scores)

def run_multi_sample(model, dataset):
    scores = None
    w_array = []
    h_array = []
    thresh_list = []
    for data in dataset:
        score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        if scores is None:
            scores = score
        else:
            scores = np.concatenate([scores, score], axis=0)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.array(scores), np.array(w_array), np.array(h_array), thresh_list


def calc_boundary_points(origin,x_length,y_length):
    a = (origin[0] - (x_length/2), origin[1] - (y_length/2))
    b = (a[0], a[1] + y_length)
    c = (a[0] + x_length, a[1] + y_length)
    d = (a[0] + x_length, a[1])
    return [a, b, c, d]

def make_geo(origin,x_length,y_length):
    geo = shapely.geometry.Polygon(calc_boundary_points(origin,x_length,y_length))
    return geo

def union(x):
        return unary_union(x)

def georeference_masks(masked_image,original_image,output_path,epsg,new_x,new_y,new_tform):
    dataset = rasterio.open(masked_image, 'r')
    original = rasterio.open(original_image, 'r')
    new_dataset = np.clip(dataset.read(1)+dataset.read(2)+dataset.read(3),0,255)
    bands = [1]
    data = np.reshape(new_dataset, (1,) + new_dataset.shape )
    
    crs = {'init': f'epsg:{epsg}'}
    
    if new_tform==False:
        tform = original.transform
    else:
        tform = rasterio.Affine(original.transform[0],original.transform[1],new_x,
                                original.transform[3],original.transform[4],new_y)

    with rasterio.open(output_path, 'w', driver='GTiff',
                    width=data.shape[2], height=data.shape[1],
                    count=1, dtype=data.dtype, nodata=0,
                    transform=tform, crs=crs) as dst:
        dst.write(data, indexes=bands)
    return original.get_transform()

def mask_to_shape(geo_mask,transform,n_cores,n_geometries_in_chunk,shape_destination,epsg):
    x_length = abs(transform[1])
    y_length = abs(transform[5])
    print("Reading raster...")
    mask_img = rxr.open_rasterio(geo_mask)
    x, y, mask = mask_img.x.values, mask_img.y.values, mask_img.values
    x, y = np.meshgrid(x, y)
    x, y, mask = x.flatten(), y.flatten(), mask.flatten()

    print("Converting to GeoDataFrame...")
    mask_pd = pd.DataFrame.from_dict({'mask': mask, 'x': x, 'y': y})
    mask_threshold = 0
    mask_pd = mask_pd[mask_pd['mask'] > mask_threshold]
    mask_pd['polygons'] = mask_pd.apply(lambda x: make_geo((x['x'],x['y']),x_length,y_length),axis=1)

    dem_vector = gpd.GeoDataFrame(geometry=gpd.GeoSeries(mask_pd['polygons'], crs=f'EPSG:{epsg}'))
    dem_vector['mask'] = mask_pd['mask']

    # Converting GeoSeries to list of geometries
    geoms = dem_vector['geometry'].tolist() #list(dem_vector)

    # # Converting geometries list to nested list of geometries
    geom_arr = []

    for i in range(0, len(geoms), n_geometries_in_chunk):
        geom_arr.append(geoms[i:i+n_geometries_in_chunk])

    # Creating multiprocessing pool to perform union operation of chunks of geometries
    with mp.Pool(n_cores) as p:
        geom_union = p.map(union, geom_arr)

    # Perform union operation on returned unioned geometries
    total_union = unary_union(geom_union)

    # Creating GeoDataFrame for total_union
    union_vector_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(total_union))

    # Saving GeoDataFrame to shapefile
    union_vector_gdf.to_file(shape_destination, driver='GeoJSON', crs=f'EPSG:{epsg}')
    
def main_shape_create(sheet,epsg,geo_mask_dest,masked_image_path,original_tile,n_cores,chunk_size,shape_dest,new_x=None,new_y=None,new_tform=False):
    transform = georeference_masks(masked_image_path, original_tile, geo_mask_dest,epsg,new_x,new_y,new_tform)
    mask_to_shape(geo_mask_dest,transform,n_cores,chunk_size,shape_dest,epsg)
    