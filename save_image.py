import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils import data
import torchvision
import argparse
import os
import numpy as np
import pandas as pd
import csv
import time
import sys
sys.path.append('..')
from utils import *
from attacks.subspace_attack import *
from attacks.mim_subspace_attack_opt import *
from attacks.baseline_attacks import *
import pdb
import timm
import PIL

parser = argparse.ArgumentParser(description='eval attacks in PyTorch')

parser.add_argument('--input_dir', default="/data/public/data/nipsgame2017_dataset", type=str, help='the direction of clean image')
parser.add_argument('--input_csv', default="/data/public/data/nipsgame2017_dataset_py/dev_dataset.csv", type=str, help='the direction of clean image csv ')
parser.add_argument('--attack_name', default='mim_sub', type=str, help='attack_name')
parser.add_argument('--beta', default=256, type=float, help='the number of beta')
parser.add_argument('--out_dir', default="/data/huangtao/projects/subsapce-attack", type=str, help='the direction of results.txt')
parser.add_argument('--source_model', default="resnet_v2_50", type=str, help='the source model which crafted AE')
parser.add_argument('--target_model', default="resnet_v2_50", type=str, help='the target_model which test the black-box attack')

parser.add_argument('--device', default=0, type=int, help='gpu device')
parser.add_argument('--use_cuda', default=True, type=bool, help='use gpu or not')
parser.add_argument('--epsilon', default=16/255, type=float, help='the number of epsilon')
parser.add_argument('--iter_num', default=100, type=int, help='the number of iter_num')
parser.add_argument('--decay_factor', default=1, type=float, help='the number of decay_factor')
parser.add_argument('--model_dir', default="/data/huangtao/models/torch_net_weight/", type=str, help='the target_model which test the black-box attack')
parser.add_argument('--Tpt', default=10, type=int, help='the parameter to control the variance of the initial Guassion distribution')
parser.add_argument('--N', default=10, type=int, help='the number of edges of simplex')
parser.add_argument('--S', default=5, type=int, help='the number of share steps')
parser.add_argument('--n', default=5, type=int, help='the iter_number of mim in al2')
parser.add_argument('--lr', default=0.01, type=float, help='the learning rate of optimization problem')
parser.add_argument('--N_vt', default=15, type=int, help="the simpling number of variance tuning")
parser.add_argument('--beta_vt', default=1.5, type=float, help='')
parser.add_argument('--save_image_dir', default='/data/huangtao/projects/subsapce-attack/perturbed_imagenet', type=str, help='')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device % 8)


def save_img(save_path, img, split_channel=False):
    img_ = np.array(img * 255).astype('uint8')
    if split_channel:
        for i in range(img_.shape[2]):
            ch_path = save_path + "@channel{}.jpg".format(i)
            ch = Image.fromarray(img_[:, :, i])
            ch.save(ch_path)
    else:
        Image.fromarray(img_,mode='RGB').save(save_path)

# save_img(os.path.join(save_path, origin_file[k]), advs[k].detach().permute(1, 2, 0).cpu())

def save_image(test_loader,device,eps,source_model,attack_name,iter_num,beta,decay_factor,N,N_vt,beta_vt):    
    iter = 0
    criterion = nn.NLLLoss()
    # pdb.set_trace()
    for data,name,target,TargetClass in test_loader:
        # target = target - 1
        iter+=1
        data,target=data.to(device),target.to(device)
        if attack_name=="mim_line":
            perturbed_data=mim_line_attack(data,eps,iter_num,source_model,decay_factor,target,beta,criterion,device)            
        if attack_name=="mim_simplex_attack":
            perturbed_data=mim_simplex_attack(data,eps,iter_num,source_model,decay_factor,target,beta,criterion,device,N)            
        if attack_name=="mim":
            perturbed_data=mim_attack(data,eps,iter_num,source_model,criterion,decay_factor,target,device)
        if attack_name=="mim_ens":
            perturbed_data=mim_ens_attack(data,eps,iter_num,source_model,criterion,target,beta,decay_factor,device,Tpt,N)
        if attack_name=="bim":
            perturbed_data=bim_attack(data,eps,iter_num,source_model,criterion,target,device)
        if attack_name=="vmim":
            perturbed_data=vmim_attack(data, eps, iter_num, source_model, criterion, decay_factor, target, device, N_vt, beta_vt)
        if attack_name=="vmim_ens":
            perturbed_data=vmim_ens_attack(data, eps, iter_num, source_model, criterion, target, decay_factor, device, N, N_vt, beta_vt)
        if attack_name=="vmim_simplex_attack":
            perturbed_data=vmim_simplex_attack(data, eps, iter_num, source_model, decay_factor, target, beta, criterion,device, N,N_vt,beta_vt)
        if attack_name=="vnim_simplex_attack":
            perturbed_data=vnim_simplex_attack(data, eps, iter_num, source_model, decay_factor, target, beta, criterion,device, N,N_vt,beta_vt)
        if attack_name=="vnim":
            perturbed_data=vnim_attack(data, eps, iter_num, source_model, criterion, decay_factor, target, device)
        if torch.max(perturbed_data) > 1+10e-4 or -torch.max(-perturbed_data) < 0:
            print("Illegal image!!\nmax is {:.5f}\nmin is {:.5f}".format(torch.max(perturbed_data),-torch.max(-perturbed_data)))
            break
        if torch.max(perturbed_data - data) > 0.063 or -torch.max(data - perturbed_data) < -0.063:
            print("Eps error!!\nmax is {:.3f}\nmin is {:.3f}".format(torch.max(perturbed_data - data),-torch.max(data -perturbed_data)))
            break
        if torch.isnan(perturbed_data.max()) or torch.isnan(perturbed_data.min()):
            print("nanerror!!")
            break
        if iter == 2:
            break
        save_img(os.path.join(args.save_image_dir,'adv')+name[0], perturbed_data.reshape(3,299,299).permute(1,2,0).detach().cpu())
    return 
def main():
    ##输出实验参数设置
    print('attack_name={}'.format(args.attack_name))
    print('N={}'.format(args.N))
    print('beta={}'.format(args.beta))
    print('beta_vt={}'.format(args.beta_vt))
    print('source_model={}'.format(args.source_model))


    # data_loader
    preprocess = transforms.Compose([
    transforms.ToTensor()])
    clean_dataset = NIPS_GAME(args.input_dir, args.input_csv, preprocess)
    test_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False,  pin_memory=True)
    # use cuda
    device=torch.device('cuda:{}'.format(str(args.device % 8)) if args.use_cuda and torch.cuda.is_available else 'cpu')
    # load pretrain_models
    if args.source_model in ['resnet_v2_152','resnet_v2_101','inception_v3',"resnet_v2_50"]:
        source_model = get_model(args.source_model,args.model_dir) 
    else:
        source_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    timm.create_model(args.source_model, pretrained=True))
    if args.target_model in ['resnet_v2_152','resnet_v2_101','inception_v3',"resnet_v2_50"]:
        target_model = get_model(args.target_model,args.model_dir) 
    else:
        target_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    timm.create_model(args.target_model, pretrained=True))
    source_model, target_model = source_model.to(device).eval(), target_model.to(device).eval()
    # process images
    save_image(test_loader,device,args.epsilon,source_model,args.attack_name,args.iter_num,args.beta,args.decay_factor,args.N,args.N_vt,args.beta_vt)

if __name__ == '__main__':
    main()


