import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import metrix,annotation_process

class CSRA_Trainer(object):
    def __init__(self, opt, net):
        self.net = net
        self.opt = opt
        self._para_initailization()

    def _para_initailization(self):
        self.criterion = nn.CrossEntropyLoss()
        if self.opt.mode == 'train':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr, betas=(0.9,0.99),weight_decay=self.opt.weight_decay)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_AUC = -1
        self.max_acc = -1
        self.max_vmix = -1
        self.epoch = 0

    def adjust_learning_rate(self,decay):
        lr_list = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)

    # loss for mutil-task
    def multi_loss(self,mp,tl):
        loss1 = self.criterion(mp[:,:3], torch.argmax(tl[:,:3],dim=1))
        loss2 = self.criterion(mp[:,3:6], torch.argmax(tl[:,3:6],dim=1))
        loss3 = self.criterion(mp[:,6:9], torch.argmax(tl[:,6:9],dim=1))
        loss4 = self.criterion(mp[:,9:], torch.argmax(tl[:,9:],dim=1))
        loss = (loss1 + loss2 + loss3 + 2*loss4) / 5
        return loss

    def train(self, train_loader):
        self.net.train()
        ###########  parameters of metrics  ###############
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []
        l_mean_acc = []
        l_single_acc = []
        acc_print_precision = 4

        self.epoch += 1
        train_loader = tqdm(train_loader, desc='Training')

        ########### training ###########
        for i, (input1, input2, filenames, targets) in enumerate(train_loader):
            device = self.device
            pm_target = targets.numpy()
            if self.opt.use_gpu:
                imgs = input1.to(device=device, dtype=torch.float32)
                masked_imgs = input2.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
            else:
                imgs = input1.type(torch.float32)
                masked_imgs = input2.type(torch.float32)
                targets = targets.type(torch.float32)

            pred, cam1, cam2, ds_region = self.net(imgs, masked_imgs, targets, 'train')    #forward
            loss1 = self.multi_loss(pred,targets)
            loss2 = self.CSRA_loss(cam1, cam2, ds_region,self.opt.ap_size)
            loss = loss1 + 0.1 * loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##################  metrics evaluation  #################
            pm_pred = pred.cpu().detach().numpy()
            epoch_loss1.append(loss1.item())

            if loss2 == 0:
                epoch_loss2.append(0)
            else:
                epoch_loss2.append(loss2.item())
            epoch_loss.append(loss.item())

            if (i+1)%1==0:
                pm_pred = annotation_process.pred2multihot(pm_pred)
                single_acc = metrix.cal_acc_single(pm_target,pm_pred)
                l_mean_acc.append(np.mean(single_acc[[0,1,2,3]]))
                l_single_acc.append(single_acc)
                train_loader.set_description(f"loss: {round(np.mean(epoch_loss),6)},loss1: {round(np.mean(epoch_loss1),6)},\
                loss2: {round(np.mean(epoch_loss2),6)}, mean_acc: {np.around(np.mean(l_mean_acc),decimals=acc_print_precision)}, \
                single_acc: {np.around(np.mean(l_single_acc,axis=0),decimals=acc_print_precision)}")

        return np.mean(l_single_acc,axis=0)[-1]

    def eval(self,val_loader):
        self.net.eval()
        opt = self.opt
        device = self.device
        f_pred = []
        f_gt = []
        val_loader = tqdm(val_loader, desc='Validation')
        for i, (input1, input2, filenames, targets) in enumerate(val_loader):
            if opt.use_gpu:
                imgs = input1.to(device=device, dtype=torch.float32)
                masked_imgs = input2.to(device=device, dtype=torch.float32)
            else:
                imgs = input1.type(torch.float32)
                masked_imgs = input2.type(torch.float32)
            pred = self.net(imgs,masked_imgs,None,'val')
            pm_pred = pred.cpu().detach().numpy()

            if f_pred == []:
                f_pred = pm_pred[:,9:]
                f_gt = targets.numpy()[:,9:]
            else:
                f_pred = np.append(f_pred,pm_pred[:,9:],axis=0)
                f_gt = np.append(f_gt,targets.numpy()[:,9:],axis=0)

        AUC = metrix.cal_AUC(f_gt,f_pred)
        F1,N_F1 = metrix.cal_F1(f_gt,f_pred)
        PPV,NPV = metrix.cal_PV(f_gt,f_pred)
        ACC = metrix.cal_ACC(f_gt,f_pred)
        vmix = (AUC+N_F1) / 2
        info = f'[epoch {self.epoch}], vmix: {round(vmix,6)}, AUC: {round(AUC,4)}, PPV: {round(NPV,4)}, NPV: {round(PPV,4)}, F1: {round(N_F1,4)}, ACC: {round(ACC,4)}'
        print(info)

        if opt.mode == 'test':
            return info
        elif not opt.is_debug:  # save model
            if self.max_vmix < vmix:
                if self.max_acc < ACC:
                    self.max_acc = ACC
                if self.max_AUC < AUC:
                    self.max_AUC = AUC
                self.max_vmix = vmix
                try:
                    os.mkdir(opt.model_save_path)
                except OSError:
                    pass
                torch.save(self.net.state_dict(),
                           os.path.join(opt.model_save_path, 'CP_epoch{}_vmix_{}.pth'.format(self.epoch,round(vmix, 4))))

    ''' in CSRA_loss module:
        0. this module is wiht respect to algorithm in Table 1.
        1. GCAM1,GCAM2,th1,th2 are with respect to the paper, GCAM1,GCAM2 are arrays contain all classes
        2. region mask describe the region of bulbar conjunctiva, it could be binary mask or gray-value mask
        3. final cos loss is respective to mean Lf for all classes
    '''
    def CSRA_loss(self,GCAM1,GCAM2,region_mask,ap_size):
        th1 = 0.8
        th2 = 0.4
        index0 = GCAM1.shape[0]*GCAM1.shape[1]
        GCAM1 = GCAM1.reshape([index0,-1])
        GCAM2 = GCAM2.reshape([index0,-1])
        region_mask = region_mask.reshape([index0,-1]) / ap_size
        count = 0
        cos_loss = 0
        for i in range(index0):
            sub_loss1 = 1 - torch.cosine_similarity(GCAM1[i],region_mask[i],dim=0)
            sub_loss2 = 1 - torch.cosine_similarity(GCAM2[i],region_mask[i],dim=0)
            if sub_loss1 < th1 and sub_loss2 < th1:
                continue
            else:
                if sub_loss1 > sub_loss2:
                    sub_loss = 1 - torch.cosine_similarity(GCAM1[i],GCAM2[i].clone().detach(),dim=0)
                else:
                    sub_loss = 1 - torch.cosine_similarity(GCAM1[i].clone().detach(),GCAM2[i],dim=0)
                if sub_loss < th2:
                    continue
                if count == 0:
                    cos_loss = sub_loss
                else:
                    cos_loss = cos_loss + sub_loss
                count += 1
        if count > 0:
            cos_loss = cos_loss / count

        return cos_loss
