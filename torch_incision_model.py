# import time

# import torch.nn.functional as F
# from torchvision.io import read_image
# from torch.utils.data import Dataset, DataLoader, random_split
# import torch.optim as optim
# import albumentations as A

import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import resnet18, ResNet18_Weights

# from dataset import *

backbone_out = 512*4*7
# define model
class Model1(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        self.fc1 = nn.Linear(backbone_out, 512) # intermediate layer between predictons
        self.fc_points_out = nn.Linear(512, 32) # 96 coordinates for 16 incisions + 16 stitches
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        x = nn.functional.relu(self.pretrained(x)) # shape (512, 4, 7) here (for (3,128,255))
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc_points_out(x))
        return x
    
class Model11(nn.Module):
    def __init__(self, pretrained_model=None):
        super().__init__()
        if pretrained_model is not None:
            backbone = list(pretrained_model.children())[0]
            self.pretrained = backbone
        else:
            self.pretrained = nn.Conv2d(3,3)
        self.fc1 = nn.Linear(backbone_out, 1024) # intermediate layer between predictons
        self.fc_points_out = nn.Linear(1024, 96) # 96 coordinates for 16 incisions + 16 stitches
        self.fc_objectness = nn.Linear(1024, 16) # 1 value / stitch
        self.objectness_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)
        # self.maxpool = nn.MaxPool2d(5, stride=2) 
    
    def forward(self, x):
        x = nn.functional.relu(self.pretrained(x)) # shape (512, 4, 7) here (for (3,128,255))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        p = torch.tanh(self.fc_points_out(x))
        o = self.objectness_activation(self.fc_objectness(x))
        return p, o
    # load pretrained vgg + add its first module as conv feature backbone: 
    # weights = VGG11_Weights.DEFAULT
    # pretrained = vgg11(weights=weights)
    # model = Model1(pretrained_model=pretrained)


def load_incision_model(path):
    '''loads .pth checkpoint file given its path'''
    checkpoint = torch.load(path, map_location='cpu')
    backbone = vgg11()
    backbone = list(backbone.children())[0]
    model_loaded = Model1(backbone)
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.eval()
    model_loaded.to('cpu')

    return model_loaded


def run_inference(model, image):
    ''' In: image of size 128x255 
        out: 32 incision points coordinates '''
    # image = torch.tensor(image).permute(2,1,0)
    # image = nn.functional.interpolate(image, size=(128,255))
    # model.eval()
    im = (image/128.0) - 1
    # im2 = (im.unsqueeze(0)).to('cpu')
    points_pred = model(im)
    points_pred = points_pred[0]
    print(f"points_pred shape: {points_pred.shape}")
    points_pred = points_pred.reshape([16,2]).detach().cpu()
    points_pred += 1
    points_pred[:,0] *= (255.0 /2)
    points_pred[:,1] *= (128.0 /2)
    return points_pred.numpy()

