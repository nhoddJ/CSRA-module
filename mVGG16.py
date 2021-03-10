import torch
import torch.nn as nn
import sys
sys.path.append('../')
import torch.nn.functional as F
import bw_module as bwm

class CSRA_VGG16_2branch(nn.Module):
    def __init__(self, model1, model2,opt, device, init_weights=True):
        super(CSRA_VGG16_2branch, self).__init__()
        foot = 2048
        self.device = device
        self.opt = opt
        self.features1 = model1.features
        self.features2 = model2.features
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            #nn.Dropout()
        )
        self.classifier11 = nn.Conv2d(512, foot, kernel_size=1, padding=0,bias=False)
        self.classifier12 = nn.Conv2d(512, foot, kernel_size=1, padding=0,bias=False)
        self.classifier2 = nn.Sequential\
            (
             nn.ReLU(True),
             #nn.Dropout(),
             nn.Conv2d(foot, opt.n_class, kernel_size=1, padding=0,bias=False)
             )
        self.bw_relu = bwm.bw_relu()

        if init_weights:
            self._initialize_weights()

    def forward(self, x1,x2,gt,mode):
        x1 = self.features1(x1)
        x2m = self.features2(x2)
        x11 = self.avgpool(x1)
        x22 = self.avgpool(x2m)
        x11= self.classifier11(x11)
        x22 = self.classifier12(x22)
        x = x11 + x22
        relux = self.classifier2[0](x)
        self.relu_mask = (relux > 0).detach().float()
        x = self.classifier2[1](relux)
        x = torch.squeeze(x,dim=2)
        x = torch.squeeze(x,dim=2)

        if mode == 'train':
            cam1, cam2  = self.get_forward_GradCAM_maps(x1,x2m)  # get forward CAM maps of all classes of two branches
            region_mask = self.mask_downsampling_v2(x2,(x2m.shape[-2],x2m.shape[-1]))
            return x,cam1.reshape(cam1.shape[0],cam1.shape[1],-1),cam2.reshape(cam1.shape[0],cam1.shape[1],-1),region_mask
        else:
            return x

    def get_forward_GradCAM_maps(self,fm1,fm2):
        if self.opt.use_gpu:
            temp = torch.ones((fm1.shape[0],self.opt.n_class,1,1),dtype=torch.float32).to(device=self.device)
        else:
            temp = torch.ones((fm1.shape[0],self.opt.n_class,1,1),dtype=torch.float32)

        # forward CAM
        for i in range(temp.shape[1]):
            w2t = nn.functional.conv2d(torch.unsqueeze(temp[:,i,:,:],dim=1),torch.unsqueeze(self.classifier2[1].weight[i],dim=1))
            w2t = self.bw_relu(w2t,self.relu_mask)
            cam_weight1 = nn.functional.conv2d(w2t,self.classifier11.weight.permute(1,0,2,3))
            cam_weight2 = nn.functional.conv2d(w2t,self.classifier12.weight.permute(1,0,2,3))

            if i == 0:
                cam1 = torch.sum(fm1 * cam_weight1/self.opt.ap_size, dim=1,keepdim=True)
                cam2 = torch.sum(fm2 * cam_weight2/self.opt.ap_size, dim=1,keepdim=True)
            else:
                cam1 = torch.cat((cam1,torch.sum(fm1 * cam_weight1/self.opt.ap_size, dim=1,keepdim=True)),dim=1)
                cam2 = torch.cat((cam2,torch.sum(fm2 * cam_weight2/self.opt.ap_size, dim=1,keepdim=True)),dim=1)

        return cam1,cam2

    #binary region mask with mean-pooling for all classes of all tasks
    def mask_downsampling(self,imgs,fm_shape):
        imgs = imgs.clone().detach()
        unit_shape = (224//fm_shape[0],224//fm_shape[1])
        imgs = torch.unsqueeze(imgs[:,0,:,:] * 0.299 + imgs[:,1,:,:] * 0.587 + imgs[:,2,:,:] * 0.114,dim=1)
        imgs[imgs>0] = 1
        imgs = F.avg_pool2d(imgs,kernel_size=(unit_shape[0],unit_shape[1]),stride=(unit_shape[0],unit_shape[1]))

        for i in range(imgs.shape[0]):
            imgs[i] = imgs[i] - torch.mean(imgs[i])
        imgs = imgs.repeat(1,11,1,1)

        return imgs.reshape(imgs.shape[0],imgs.shape[1],-1)

    # task illumination using masked image but not binary region mask for downsampling, it emphasizes gray-scale value
    def mask_downsampling_v2(self,imgs,fm_shape):
        ds_mode = ['g','g','g','r','r','r','r','r','r','r','r']
        imgs = imgs.clone().detach()
        unit_shape = (224//fm_shape[0],224//fm_shape[1])
        gray = torch.unsqueeze(imgs[:,0,:,:] * 0.299 + imgs[:,1,:,:] * 0.587 + imgs[:,2,:,:] * 0.114,dim=1)
        ds_gray = F.avg_pool2d(gray,kernel_size=(unit_shape[0],unit_shape[1]),stride=(unit_shape[0],unit_shape[1])) * (unit_shape[0]*unit_shape[1])
        gray[gray>0] = 1
        ds_region = F.avg_pool2d(gray,kernel_size=(unit_shape[0],unit_shape[1]),stride=(unit_shape[0],unit_shape[1]))
        ds_gray = ds_gray / (ds_region+1e-9)

        for i in range(imgs.shape[0]):
            ds_region[i] = ds_region[i] - torch.mean(ds_region[i])
            ds_gray[i] = ds_gray[i] - torch.mean(ds_gray[i])
        ds_masks = ds_region.repeat(1,self.opt.n_class,1,1)
        ds_gray = torch.squeeze(ds_gray,dim=1)

        for i in range(self.opt.n_class):
            if ds_mode[i] == 'g':
                ds_masks[:,i] = ds_gray

        return ds_masks

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

