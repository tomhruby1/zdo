import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import copy
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

COLOR = {
    'Incision': 'red',
    'Stitch': 'green',
    'incision': 'red',
    'stitch': 'green'
}

MEAN = np.array([166.25210137, 128.91363988, 131.47121054])
STD = np.array([47.85307378, 43.52442252, 44.18917168])

class ZdoDataset(Dataset): 
    def __init__(self, data_json_p, images_p=Path('data/images/default'), 
                 image_size=(128, 255), transform=None, normalize=True):
        '''Dataset instance, ima.ges path expected
            args: 
                - image_size: target size to which all images will be interpolated
                - transform: any other transformation to be applied'''
        
        with open(data_json_p, 'r') as f:
            data = json.load(f)
        self.images_p = images_p
        self.transform = transform
        # only couple small images - all will be loaded into memory
        self.data, self.images = interpolate_to_size(data, size=image_size)
        self.normalize = normalize
        # in x,y shape ... h,w != x,y
        self.image_size = np.asarray([image_size[1], image_size[0]], dtype=float) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        im = self.images[image_id]
        # TODO: only incision for now
        points = np.array(self.data[image_id]['incision'], dtype=float)
        
        if self.transform is not None:
            transformed = self.transform(image=im, keypoints=points)
            im = transformed['image']
            points = transformed['keypoints']

        if self.normalize: # [0,255] => [-1,1]
            im = (im/128.0) - 1 # image
            points = points / (self.image_size/2) - 1
    
        return torch.tensor(im).permute(2,0,1), torch.tensor(points)
    
    def get_raw_item(self, image_id):
        im = self.images[image_id]
        # TODO: only incision for now
        points = np.array(self.data[image_id]['incision'], dtype=float)
        
        return im, np.array(points)

def interpolate_to_size(data, size=(128,255)):
    '''Interpolates the image data to the provided size. 
        Annotations are transformed accordingly.'''
    images_interpolated = {} #interpolated image tensors
    data_new = copy.deepcopy(data)
    for image_id in data:

        image_p = Path('data/images/default')/data[str(image_id)]['file']

        im = read_image(str(image_p))
        inter_coeff = (size[1]/im.shape[2], size[0]/im.shape[1])  # the interpolation ratio, xy switched (h,w)

        im = im.unsqueeze(0)
        im2 = F.interpolate(im, size=size) # size=(3,128,255)
        images_interpolated[image_id] = im2[0].permute(1,2,0).numpy()

        # interpolate the annotation
        # terrible stitches resize for interpolation
        for i, st in enumerate(data[str(image_id)]['stitches']):
            for j,pt in enumerate(st):
                pt = np.array(pt, dtype=float)
                pt *= inter_coeff
                data_new[str(image_id)]['stitches'][i][j] = list(pt)
        for i, pt in enumerate(data[str(image_id)]['incision']):
            pt = np.array(pt, dtype=float)
            pt *= inter_coeff
            data_new[str(image_id)]['incision'][i] = list(pt)
    
    return data_new, images_interpolated

def visualize(image:Union[torch.tensor, np.ndarray], incision:Union[torch.tensor, np.ndarray], 
              show_points=True, unnormalize=False, imsize=(128, 255)):
    if type(image) == torch.Tensor:
        image = image.permute(1,2,0).numpy()
    if unnormalize:
        image = (image * 128.0 + 128.0).astype(int)  #(image * np.array([255,255,255]) + np.array([255,255,255]))
    fig, ax = plt.subplots()
    ax.imshow(image)

    # plot incision
    if type(incision) == torch.tensor:
        incision = incision.cpu().numpy()
    x_coords, y_coords = zip(*incision)
    x_coords = np.array(x_coords, dtype=float) 
    y_coords = np.array(y_coords, dtype=float)
    if unnormalize:
        x_coords = (x_coords+1) * float(imsize[1])/2  
        y_coords = (y_coords+1) * float(imsize[0])/2
    ax.plot(x_coords, y_coords, color=COLOR['Incision'], linewidth=2)
    
    # plot incision polyline points
    if show_points:
        for i in range(len(x_coords)):
            plt.plot(x_coords[i], y_coords[i],'xc')

def visualize_data(data, image_id, image=None, incision=None):
    if image is not None:
        if type(image) == torch.Tensor:
            if len(image.shape) == 4:
                image = image[0].permute(1, 2, 0) # TODO: what if not batched?
            else: print(f"nope..")
    else:
        image_id = str(image_id)
        image_path = Path('data/images/default')/data[image_id]['file']
        image = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # plot incision
    points = data[image_id]['incision']
    x_coords, y_coords = zip(*points)
    x_coords = np.array(x_coords, dtype=float)
    y_coords = np.array(y_coords, dtype=float)
    ax.plot(x_coords, y_coords, color='g', linewidth=2)
    
    # plot incision polyline points
    for i in range(len(x_coords)):
        plt.plot(x_coords[i], y_coords[i],'xc')
    
    # plot all the stitches
    for i in range(len(data[image_id]['stitches'])):
        points = data[image_id]['stitches'][i]
        x_coords, y_coords = zip(*points)
        x_coords = np.array(x_coords, dtype=float)
        y_coords = np.array(y_coords, dtype=float)
        ax.plot(x_coords, y_coords, color='r', linewidth=2)

def visualize_old(data, image_id):
    '''Visualizes image and it's Incission and Stitch annotations'''

    image_path = Path('data/images/default')/data[image_id]['file']
    image = mpimg.imread(image_path)
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    for idx, label in enumerate(data[image_id]['label']):
        points = data[image_id]['points'][idx]
        x_coords, y_coords = zip(*points)
        x_coords = np.array(x_coords, dtype=float)
        y_coords = np.array(y_coords, dtype=float)
        ax.plot(x_coords, y_coords, color=COLOR[label], linewidth=2)
    # Show the plot
    plt.show()
