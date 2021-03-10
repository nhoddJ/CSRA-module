import numpy as np
import os
import random
import cv2
import copy
import torch.utils.data as data

class basic_dataset(data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def _initialization(self):
        self.batch_pos = 0
        self.global_transform = self.opt.global_transform
        self.global_transform_xval = self.opt.global_transform_xval
        self.global_transform_para = self.opt.global_transform_para

    ############# transform ##############
    def mask(self,img,mask_dir):
        mask = cv2.imread(os.path.join(mask_dir,self.current_annotation_name[:-4]+'.bmp'))
        if len(img.shape) > len(mask.shape):
            for i in range(img.shape[-1]):
                img[:,:,i] = np.where(mask==255,img[:,:,i],np.zeros_like(img[:,:,i]))
        return img

    def resize(self, img, new_size):
        return cv2.resize(img, new_size)

    def div(self,img,c):
        return img / c


