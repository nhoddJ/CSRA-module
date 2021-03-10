import torch
import numpy as np
import os
import basic_dataset
import annotation_process
import torch.utils.data as data
import cv2

class LXD_dataset(basic_dataset.basic_dataset):
    def __init__(self,opt, data_dir,annotations,mode):
        self.opt = opt
        self.data_dir = data_dir
        self.annotations = annotations
        self.mode = mode
        print('\033[1;35m' + '#' * 20 + ' data_preprocess ' + '#' * 20 + '\033[0m')
        self._initialization()
        self.num_of_data = len(self.annotations)

    def return_batch_names(self,batch_size):
        annotaions = self.annotations[self.index[self.batch_pos-batch_size:self.batch_pos]]
        annotaions = annotaions[:,0]
        return annotaions

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, item):
        suffix = '.bmp'
        annotation = self.annotations[item]
        self.current_annotation_name = annotation[0]
        img = cv2.imread(os.path.join(self.data_dir,annotation[0][:-4]+suffix))
        if len(img.shape) == 2:  #for gray-scale image
            img = np.expand_dims(img,axis=2)

        for transform, transform_para in zip(self.global_transform,  self.global_transform_para):
            if (self.mode == 'val' or self.mode == 'test')  and transform in self.global_transform_xval :
                continue
            img = getattr(self, transform, 'None')(img,  transform_para)

        image = np.array(img)
        annotation = annotation_process.convert2multihot(annotation)

        return torch.from_numpy(np.transpose(image,(2,0,1))),self.current_annotation_name,torch.from_numpy(annotation.astype('int32'))

class con_dataset(data.Dataset):
    def __init__(self,dataset1,dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return self.dataset1.num_of_data

    def __getitem__(self, item):
        data1 = self.dataset1[item]
        data2 = self.dataset2[item]
        return data1[0],data2[0],data1[1],data1[2]