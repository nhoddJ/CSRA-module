import numpy as np
import sklearn.metrics as metrics
import torch.nn.functional as F
import torch

# calculate accuracy or each task
def cal_acc_single(np1,np2):
    label_range = [[0,3],[3,6],[6,9],[9,11]]
    acc = []
    if isinstance(np1,list):
        np1 = np.array(np1)
    if isinstance(np2,list):
        np2 = np.array(np2)
    for n1,n2 in zip(np1,np2):
        t_acc = []
        for i in range(len(label_range)):
            t1 = n1[label_range[i][0]:label_range[i][1]]
            t2 = n2[label_range[i][0]:label_range[i][1]]
            if (t1 == t2).all():
                t_acc.append(1)
            else:
                t_acc.append(0)
        acc.append(t_acc)
    acc = np.array(acc)
    if acc == []:
        return 0
    else:
        return np.mean(acc,axis=0)

def cal_AUC(y_true,y_pred):
    y_true = np.argmax(y_true,axis=1)
    y_pred_prob = F.softmax(torch.tensor(y_pred),dim=1).numpy()
    y_pred_prob = y_pred_prob[:,1]
    AUC = metrics.roc_auc_score(y_true,y_pred_prob)
    return AUC

def cal_F1(y_true,y_pred):
    y_true = np.argmax(y_true,axis=1)
    y_pred = np.argmax(y_pred,axis=1)
    F1 = metrics.f1_score(y_true,y_pred)
    N_F1 = metrics.f1_score(1-y_true,1-y_pred)
    return F1, N_F1

def cal_ACC(y_true,y_pred):
    y_true = np.argmax(y_true,axis=1)
    y_pred = np.argmax(y_pred,axis=1)
    return np.sum(y_true==y_pred) / len(y_true)

def cal_PV(y_true,y_pred):
    y_true = np.argmax(y_true,axis=1)
    y_pred = np.argmax(y_pred,axis=1)
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    TP = np.sum((y_true==1) & (y_pred == 1))
    TN = np.sum((y_true==0) & (y_pred == 0))
    return np.sum(TP) / P, np.sum(TN) / N

