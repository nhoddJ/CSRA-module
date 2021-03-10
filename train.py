import sys
extra_paths = ['trainer','dataiter','toolbox','options']
[sys.path.append(i) for i in extra_paths]
import torchvision.models as models
import mVGG16
from dataiter import LXD_dataset
from trainer import CSRA_trainer
import numpy as np
import torch
import copy
import os
import pandas as pd
from train_options import TrainOptions

def train():
    ##############   initialization   ################
    ### load csv files of annotations, initialize options
    dataroot = './data'
    train_list = np.array(pd.read_csv(os.path.join(dataroot,'labels/training_set_anno.csv')))
    val_list = np.array(pd.read_csv(os.path.join(dataroot,'labels/validation_set_anno.csv')))
    opt_mask = TrainOptions().parse()
    opt = copy.deepcopy(opt_mask)
    opt.global_transform = opt.global_transform[1:]
    opt.global_transform_para = opt.global_transform_para[1:]
    train_dataroot = os.path.join(dataroot,'dataset/train')
    val_dataroot = os.path.join(dataroot,'dataset/val')

    # define dataset and dataloader
    train_dataset = LXD_dataset.LXD_dataset(opt, train_dataroot, train_list,'train')
    train_dataset_mask = LXD_dataset.LXD_dataset(opt_mask, train_dataroot, train_list,'train')
    con_dataset_train = LXD_dataset.con_dataset(train_dataset,train_dataset_mask)
    val_dataset = LXD_dataset.LXD_dataset(opt, val_dataroot, val_list,'val')
    val_dataset_mask = LXD_dataset.LXD_dataset(opt_mask, val_dataroot, val_list,'val')
    con_dataset_val = LXD_dataset.con_dataset(val_dataset,val_dataset_mask)
    train_loader = torch.utils.data.DataLoader(con_dataset_train, batch_size=opt.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(con_dataset_val, batch_size=4, shuffle=False)

    # define model and trainer
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    model1 = models.vgg16_bn(pretrained=False)
    model2 = models.vgg16_bn(pretrained=False)
    model =  mVGG16.CSRA_VGG16_2branch(model1,model2,opt,device)
    if opt.use_gpu:
        model.to(device=device)
    trainer = CSRA_trainer.CSRA_Trainer(opt,model)

    # training and validation
    for epoch in range(opt.epochs):
        acc = trainer.train(train_loader)
        with torch.no_grad():
            trainer.eval(val_loader)
        if acc > 0.98:
            break
        if epoch == 10:
            trainer.adjust_learning_rate(0.1)

if __name__ == '__main__':
    train()
