import os.path
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import json
import utils
import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast

def load_nii(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

class BraTS2021(Dataset):
    def __init__(self, json_path, data_path , mode = 'train', val_count = 20):
        super(BraTS2021, self).__init__()
        self.mode = mode
        self.data_path = data_path

        with open(json_path,'r') as f:
            self.data = json.load(f)
        self.data_name = []
        if mode == 'train':
            for patient in self.data['training']:
                if patient['fold'] != 2 and patient['fold'] != 1:
                    self.data_name.append(patient)
        elif mode == 'val':
            for patient in self.data['training']:
                if patient['fold'] == 2:
                    self.data_name.append(patient)
                    if len(self.data_name) == val_count:
                        return
        elif mode == 'test':
            for patient in self.data['training']:
                if patient['fold'] == 1:
                    self.data_name.append(patient)

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, item):
        patient = self.data_name[item]
        img = np.zeros(shape=(4, 160, 240, 240))
#        img = np.zeros(shape=(4, 155, 240, 240))
        for i in range(4):
            img[i, 0:155, :, :] = load_nii(os.path.join(self.data_path, patient['image'][i]))
        img = torch.tensor(img)

        label = load_nii(os.path.join(self.data_path, patient['label']))
        label = torch.tensor(label.astype('int32'))
        et = label == 4
        tc = torch.logical_or(label == 1, label == 4)
        wt = torch.logical_or(tc, label == 2)
        label = torch.stack([et, tc, wt])

        img ,label = utils.regulate_data(img, label, target_size=(128,128,128), mode=self.mode)
        img = img.float()
        label = label.float()
        if self.mode == "test":
            return img, label, patient['image'][0]
        else:
            return img, label





class DataAugmenter(nn.Module):
    def __init__(self):
        super(DataAugmenter,self).__init__()
        self.flip_dim = []
        self.zoom_rate = uniform(0.7, 1.0)
        self.sigma_1 = uniform(0.5, 1.5)
        self.sigma_2 = uniform(0.5, 1.5)
        self.image_zoom = Zoom(zoom=self.zoom_rate, mode="trilinear", padding_mode="constant")
        self.label_zoom = Zoom(zoom=self.zoom_rate, mode="nearest", padding_mode="constant")
        self.noisy = RandGaussianNoise(prob=1, mean=0, std=uniform(0, 0.33))
        self.blur = GaussianSharpen(sigma1=self.sigma_1, sigma2=self.sigma_2)
        self.contrast = AdjustContrast(gamma=uniform(0.65, 1.5))
    def forward(self, images, lables):
        with torch.no_grad():
            for b in range(images.shape[0]):
                image = images[b].squeeze(0)
                lable = lables[b].squeeze(0)
                if random() < 0.15:
                    image = self.image_zoom(image)
                    lable = self.label_zoom(lable)
                if random() < 0.5:
                    image = torch.flip(image, dims=(1,))
                    lable = torch.flip(lable, dims=(1,))
                if random() < 0.5:
                    image = torch.flip(image, dims=(2,))
                    lable = torch.flip(lable, dims=(2,))
                if random() < 0.5:
                    image = torch.flip(image, dims=(3,))
                    lable = torch.flip(lable, dims=(3,))
                if random() < 0.15:
                    image = self.noisy(image)
                if random() < 0.15:
                    image = self.blur(image)
                if random() < 0.15:
                    image = self.contrast(image)
                images[b] = image.unsqueeze(0)
                lables[b] = lable.unsqueeze(0)
            return images, lables



