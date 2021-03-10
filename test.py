import torch
import numpy as np
import os
import sys
extra_paths = ['trainer','dataiter','toolbox','options']
[sys.path.append(i) for i in extra_paths]
from test_options import TestOptions
import copy
import pandas as pd
import LXD_dataset
import mVGG16
import CSRA_trainer
import torchvision.models as models
import natsort

def ttest():
    dataroot = './data'
    test_list = np.array(pd.read_csv(os.path.join(dataroot,'labels/test_set_anno.csv')))
    opt_mask = TestOptions().parse()
    opt = copy.deepcopy(opt_mask)
    opt.global_transform = opt.global_transform[1:]
    opt.global_transform_para = opt.global_transform_para[1:]
    results_savename = os.path.join(opt.model_load_path,'results.txt')
    test_dataroot = os.path.join(dataroot,'dataset/test')
    test_set = LXD_dataset.LXD_dataset(opt, test_dataroot, test_list,'test')
    test_set_masked = LXD_dataset.LXD_dataset(opt_mask, test_dataroot, test_list,'test')
    con_sets_for_test = LXD_dataset.con_dataset(test_set,test_set_masked)
    test_loader = torch.utils.data.DataLoader(con_sets_for_test, batch_size=4, shuffle=False)
    model1 = models.vgg16_bn(pretrained=False)
    model2 = models.vgg16_bn(pretrained=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mVGG16.CSRA_VGG16_2branch(model1,model2,opt,device)
    results = []
    if opt.use_gpu:
        model.to(device=device)
    model_names = natsort.natsorted(os.listdir(opt.model_load_path))

    for model_name in model_names:
        if model_name[:3] == 'CP_':
            with torch.no_grad():
                model.eval()
                model.load_state_dict(torch.load(os.path.join(opt.model_load_path,model_name)))
                trainer = CSRA_trainer.CSRA_Trainer(opt,model)
                result = trainer.eval(test_loader)
                results.append('['+model_name+']'+' '+result)

    with open(results_savename,'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')

if __name__ == '__main__':
    ttest()
