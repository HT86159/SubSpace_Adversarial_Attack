import torch
from torch.utils import data
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import torchvision
import time
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
sys.path.append('/data/huangtao/models')
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4, 
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
log_offset = 1e-20
det_offset = 1e-6
softmax_0 = nn.Softmax(dim=0)
def clipfun(adv_image,image,eps):
    a = torch.clamp(image - eps, min=0)
    b = (adv_image >= a).float()*adv_image \
        + (adv_image < a).float()*a
    c = (b > image + eps).float()*(image + eps) \
        + (b <= image + eps).float()*b
    image = torch.clamp(c, max=1).detach()
    return image
    # minx = image - eps
    # maxx = image + eps
    # return torch.max(torch.min(perturbed_image, maxx), minx)

def cosine_similarity(tensor_a, tensor_b):
    tensor_a = tensor_a.reshape(-1, 1)
    tensor_b = tensor_b.reshape(1, -1)
    numerator = torch.matmul(tensor_b, tensor_a)
    denominator = torch.norm(tensor_b)*torch.norm(tensor_a)
    if denominator != 0:
        cosine_similarity =  numerator/ denominator
        return cosine_similarity
    else:
        return torch.tensor(1)

def get_Softmax_pre_list(ADV_image, model):
    softmax = nn.Softmax(dim=0)
    Softmax_pre = [softmax(model(adv_image)[0]).reshape(1,-1) for adv_image in ADV_image]
    return Softmax_pre

def Entropy(y):#correct tensor(n,1)--->scale
    entropy = (-y* torch.log(y + log_offset)).sum()
    return entropy

def Ensemble_Entropy(y_true, Y_pre, N, device):
    Y_pre = torch.stack(Y_pre)
    return Entropy(Y_pre.sum(0) / N)

def Ensemble_Diversity(target, Y_pre, device):
    N = len(Y_pre)
    My = torch.cat(Y_pre,0).to(device) #N * classes_num
    temp = torch.eye(My.shape[1]).to(device) #classes_num classes_num
    temp[int(target.item())-1][int(target.item())-1] = 0
    bool_R_y_true = torch.ne(torch.mm(My,temp), 0) #N * classes_num 相等是false，大部分都是true
    # print("bool_R_y_true\t",bool_R_y_true,bool_R_y_true.shape)
    mask_non_y_pred = torch.masked_select(My, bool_R_y_true) # N * class_num 验证，只有true的留下了了
    mask_non_y_pred = mask_non_y_pred.reshape(N, -1) 
    mask_non_y_pred=mask_non_y_pred/torch.norm(mask_non_y_pred,dim=1,keepdim=True) # 标准化
    matrix = torch.mm(mask_non_y_pred.T , mask_non_y_pred)
    all_log_det = torch.log(log_offset + torch.linalg.det(matrix*1e30 + det_offset) )# batch_size X 1, 1-D
    return all_log_det

def to_latent_space(x, minx, maxx):#输出就是delta+image;w-->>image;R--->(0,1)
        return minx + 0.5*(torch.tanh(x)+1)*(maxx - minx)

def to_image_space(x, minx, maxx):#image-->w(0,1);--->(-11.5129,11.5129)
        return 0.5*torch.log((x-minx)/(maxx - x))




class NIPS_GAME(data.Dataset):
    def __init__(self, dir, csv_path, transform = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path, engine='python')
        self.transform = transform

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + ".png"
        # match torch label style
        Truelabel = img_obj['TrueLabel'] 
        TargetClass = img_obj['TargetClass'] 
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')

        if self.transform:
            data = self.transform(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel, TargetClass

    def __len__(self):
        return len(self.csv)
class Normalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

def get_model(model_name, model_dir):
    model_path = os.path.join(model_dir,'tf_'+ model_name + '.npy')

    if model_name=="resnet_v2_101":
        model = tf_resnet_v2_101
    elif model_name=="resnet_v2_50":
        model = tf_resnet_v2_50
    elif model_name=="resnet_v2_152":
        model = tf_resnet_v2_152
    elif model_name=="inception_v3":
        model = tf_inception_v3
    model = nn.Sequential(
        Normalize('tensorflow'), 
        model.KitModel(model_path))
    return model

import os
def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_img(save_path, img, split_channel=False):
    img_ = np.array(img * 255).astype('uint8')
    if split_channel:
        for i in range(img_.shape[2]):
            ch_path = save_path + "@channel{}.jpg".format(i)
            ch = Image.fromarray(img_[:, :, i])
            ch.save(ch_path)
    else:
        Image.fromarray(img_,mode='RGB').save(save_path)