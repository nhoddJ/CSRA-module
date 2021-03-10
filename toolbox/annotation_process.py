import torch
import numpy as np

def Cos(x,y):
    cos = torch.matmul(x,y.view((-1,))) / ((torch.sum(x*x)+1e-9).sqrt() * torch.sum(y*y).sqrt())
    return cos

def convert2multihot(annotation):
    multihot = np.zeros(11,dtype='int32')
    anno_index = [1,2,3,4]
    anno_map = [[0,1,2],[3,4,5],[9,10],[6,7,8]]
    for i in range(4):
        x = annotation[anno_index[i]]
        multihot[anno_map[i][x]] = 1
    return multihot

def pred2multihot(annotation):
    anno_map = [[0,1,2],[3,4,5],[9,10],[6,7,8]]
    for i,a in enumerate(annotation):
        b = np.zeros_like(a,dtype='int32')
        b[anno_map[0][np.argmax(a[:3])]] = 1
        b[anno_map[1][np.argmax(a[3:6])]] = 1
        b[anno_map[2][np.argmax(a[9:])]] = 1
        b[anno_map[3][np.argmax(a[6:9])]] = 1
        annotation[i] = b
    return annotation


