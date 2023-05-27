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

MAX_STITCHES = 16

IGNORE = ['128', '180', '56']

class ZdoDataset(Dataset): 
    def __init__(self, data_json_p, images_p=Path('data/images/default'), 
                 image_size=(128, 255), transform=None, normalize=True, 
                 incision_points=-1):
        '''Dataset instance, ima.ges path expected
            args: 
                - image_size: target size to which all images will be interpolated
                - transform: any other transformation to be applied
                - incision_points: how many incision points to actually return (2...), -1 for all 16'''
        
        with open(data_json_p, 'r') as f:
            data = json.load(f)
        self.images_p = images_p
        self.transform = transform
        # only couple small images - all will be loaded into memory
        self.data, self.images = interpolate_to_size(data, size=image_size, ignore_ids=IGNORE)
        self.normalize = normalize
        # in x,y shape ... h,w != x,y
        self.image_size = np.asarray([image_size[1], image_size[0]], dtype=float) 
        
        self.number_of_points = incision_points
        if incision_points > 0:
            if incision_points % 2 == 0:
                self.number_of_points = incision_points // 2   # to be selected from start and end
            else: 
                raise Exception(f"incision_points must be divisible by 2: {incision_points} provided")  
        
        self.points = {}
        self.stitches_labels = {}
        # process all points (incision + stitches and store into one array)
        for image_id in self.data:
            incision_points = np.array(self.data[image_id]['incision'], dtype=float)
            stitches_points = np.array(self.data[image_id]['stitches'], dtype=float)
            
            stitches_labels = np.zeros(MAX_STITCHES)
            # init with somewhat random vals around the center
            stitches_points2 = np.random.rand(MAX_STITCHES, 2, 2) * 10
            N = len(stitches_points)
            if N > 0:
                for i in range(MAX_STITCHES):
                    stitches_points2[i] += np.array([[12*(i+1), 30], [12*(i+1), 80]])
            
            # distribute N stitches through the final fixed sized stitches array
            if N > 0:
                # if done randomly:
                # indices = np.arange(MAX_STITCHES)
                # np.random.shuffle(indices)
                # stitches_points2[indices[:N], :] = stitches_points
                # stitches_labels[indices[:N]] = 1
                # for i in range(MAX_STITCHES):
                #     stitch_reg_idx = 
                reg = np.array(stitches_points2)
                stitches_labels = np.zeros(MAX_STITCHES)
                # assign each stitch to the closes available reg
                for stitch_idx in range(N):
                    dist2regs = [99999]*MAX_STITCHES
                    # sort stitch regressors for each stitch by distance
                    for j in range(MAX_STITCHES):
                        dist2regs[j] = np.linalg.norm(stitches_points[stitch_idx] - reg[j])
                    closest_regs = np.argsort(dist2regs)

                    for r_id in closest_regs:
                        if stitches_labels[r_id] != 0:
                            continue
                        else: 
                            stitches_labels[r_id] = 1
                            stitches_points2[r_id, :] = stitches_points[stitch_idx]
                            break

            stitches_points2 = stitches_points2.astype('float32')
            stitches_points2 = stitches_points2.reshape([MAX_STITCHES*2, 2])

            # save stuff
            self.points[image_id] = np.vstack([incision_points, stitches_points2])
            self.stitches_labels[image_id] = stitches_labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        im, points, stitch_labels = self.get_item_by_id(image_id)
        
        return im, points, stitch_labels
        
    def get_item_by_id(self,image_id):
        im = self.images[image_id]
        points = self.points[image_id]
        if self.transform is not None:
            transformed = self.transform(image=im, keypoints=points)
            im = transformed['image']
            points = transformed['keypoints']
            # if geometrical transformation looses some points
            if self.number_of_points == -1:
                if len(np.array(points).flatten()) != 96:
                    print(f"wrong number of point coordinates ({len(np.array(points).flatten())}) for image id {image_id}")
                    raise Exception("2x16 points needed!")

        if self.normalize: # [0,255] => [-1,1]
            im = (im/128.0) - 1 # image
            points = points / (self.image_size/2) - 1
        # return as float32
        im = torch.tensor(im).permute(2,0,1).float()
        points = torch.tensor(points).float()

        # TODO: reimplement for all the points
        # if self.number_of_points > 0: 
        #     start = incision_points[:self.number_of_points, :]
        #     end = incision_points[-self.number_of_points:, :]
        #     incision_points = torch.cat([start,end])
            
        return im, points, torch.tensor(self.stitches_labels[image_id]).float()
    
    def get_raw_item(self, image_id):
        im = self.images[image_id]
        # TODO: only incision for now
        points = np.array(self.data[image_id]['incision'], dtype=float)
        
        return im, np.array(points)

def interpolate_to_size(data, size=(128,255), ignore_ids=None):
    '''Interpolates the image data to the provided size. 
        Annotations are transformed accordingly.'''
    images_interpolated = {} #interpolated image tensors
    data_new = copy.deepcopy(data)
    for image_id in data:
        if ignore_ids is not None:
            if image_id in ignore_ids:
                data_new.pop(image_id)
                continue
        
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
                # # hackfix to  ValueError: Expected x for keypoint ... to be in the range [0.0, 255], got 255.0.
                # for c in range(len(pt)):
                #     if pt[c] >= 255.0:
                #         pt[c] = 255.0 - 0.01
                data_new[str(image_id)]['stitches'][i][j] = list(pt)
        for i, pt in enumerate(data[str(image_id)]['incision']):
            pt = np.array(pt, dtype=float)
            pt *= inter_coeff
            # hackfix to  ValueError: Expected x for keypoint ... to be in the range [0.0, 255], got 255.0.
            for c in range(len(pt)):
                if pt[c] >= 255.0: pt[c] = 255.0 - 0.01
            data_new[str(image_id)]['incision'][i] = list(pt)
    
    return data_new, images_interpolated

def visualize(image:Union[torch.Tensor, np.ndarray], incision:Union[torch.Tensor, np.ndarray],
              stitches:Union[torch.Tensor, np.ndarray] = None, stitch_objectness = None,
              show_points=True, unnormalize=False, imsize=(128, 255)):
    if type(image) == torch.Tensor:
        image = image.permute(1,2,0).numpy()
    if unnormalize:
        image = (image * 128.0 + 128.0).astype(int)  #(image * np.array([255,255,255]) + np.array([255,255,255]))
    fig, ax = plt.subplots()
    ax.imshow(image)

    # print(f"image_id: {image_id}")
    # plot incision
    if type(incision) == torch.Tensor:
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
            ax.plot(x_coords[i], y_coords[i],'xc')

    # plot stitches
    if stitches is not None:
        if type(stitches) == torch.Tensor:    
            stitches = stitches.cpu().numpy()
        stitch_count = 0
        for i in range(0, len(stitches), 2):
            x_coords, y_coords = zip(*stitches[i:i+2,:])
            x_coords = np.array(x_coords, dtype=float) 
            y_coords = np.array(y_coords, dtype=float)
            alpha = 1.0
            if stitch_objectness is not None:
                alpha = float(stitch_objectness[stitch_count])
            stitch_count += 1
            if unnormalize:
                x_coords = (x_coords+1) * float(imsize[1])/2  
                y_coords = (y_coords+1) * float(imsize[0])/2
            ax.plot(x_coords, y_coords, color=COLOR['Stitch'], marker='o', alpha=alpha)
    plt.show()

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