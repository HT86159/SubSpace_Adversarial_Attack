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
from attacks.baseline_attacks import *
import pdb
import timm

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


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device % 8)


def test(test_loader,device,eps,source_model,target_model,attack_name,iter_num,beta,decay_factor,Tpt,N,S,n,lr,N_vt,beta_vt):
    acc_list=[]
    start = time.time()
    correct = 0
    success = 0
    iter = 0
    criterion =  nn.CrossEntropyLoss()
    # pdb.set_trace()
    for data,name,target,TargetClass in test_loader:
        # target = target - 1
        iter+=1
        if iter == 5:
            print(">>>>>>>>>>>>>>>>>>>>{}\t{}->{}<<<<<<<<<<<<<<<<<<<<".format(attack_name,args.source_model,args.target_model))

        data,target=data.to(device),target.to(device)
        # logits = source_model(data)[0]
        # softmax = softmax_0(logits)
        # init_target=torch.argmax(softmax,axis=0) 
        # perturbed_data = torch.zeros_like(data)
        # pdb.set_trace()
        if attack_name=="mim_sub":
            perturbed_data=mim_subspace_attack(data,eps,iter_num,source_model,criterion,target,beta,decay_factor,device,Tpt,N,lr)
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
        # if iter == 10:
        #     break
        perturbed_output = target_model(perturbed_data)[0]
        perturbed_label = perturbed_output.argmax() 
        if perturbed_label.item() == target.item():
            correct+=1
            acc_list.append(0)
        else:
            success+=1
            acc_list.append(1)
        # if iter % 500 == 0:
        #     print("atkname={}\tN={}\tbeta={}\tNvt={}\tbetavt={}\tTrain complete:{}/{}\tASR:{:.4f}".format(attack_name,N,beta,N_vt,beta_vt,iter,len(test_loader),(success)/(iter)))
        # perturbed_output_random=target_model(perturbed_data_random)[0]
        # perturbed_label_random=torch.argmax(perturbed_output_random,axis=0) 
        # if perturbed_label_random.item()==init_target.item():
        #     correct_random+=1
        # else:
        #     success_random+=1
    acc_frame = pd.DataFrame(acc_list)
    accuracy = success/iter
    # accuracy_random = success_random/len(test_loader)
    # acc_frame.to_excel("attack_name={},beta{},N{},N_vt{},beta_vt{}accuracy={:.4f}-output.xlsx".format(attack_name,beta,N,N_vt,beta_vt,accuracy))

    end=time.time()
    print("{},{}->{}beta={},N={},N_vt={},beta_vt={}\t ASR={:.4f},time:{:.4f}".format(attack_name,args.source_model,args.target_model,beta,N,N_vt,beta_vt,accuracy,end-start))
    return accuracy
def main():
    ##输出实验参数设置
    print('attack_name={}'.format(args.attack_name))
    print('N={}'.format(args.N))
    print('beta={}'.format(args.beta))
    print('N_vt={}'.format(args.N_vt))
    print('beta_vt={}'.format(args.beta_vt))
    print('source_model={}'.format(args.source_model))
    print('target_model={}'.format(args.target_model))
    print('iter_num={}'.format(args.iter_num))

    # data_loader
    preprocess = transforms.Compose([
    transforms.ToTensor()])
    clean_dataset = NIPS_GAME(args.input_dir, args.input_csv, preprocess)
    test_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False,  pin_memory=True)
    # use cuda
    device=torch.device('cuda:{}'.format(str(args.device % 8)) if args.use_cuda and torch.cuda.is_available else 'cpu')
    # load pretrain_models
    if args.source_model in ['resnet_v2_152','resnet_v2_101','inception_v3',"resnet_v2_50","inc_res_v2","adv_inception_v3" ,"ens3_adv_inc_v3", "ens4_adv_inc_v3", "ens_adv_inc_res_v2"]:
        source_model = get_model(args.source_model,args.model_dir) 
    else:
        source_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    timm.create_model(args.source_model, pretrained=True))
    if args.target_model in ['resnet_v2_152','resnet_v2_101','inception_v3',"resnet_v2_50","inc_res_v2","adv_inception_v3" ,"ens3_adv_inc_v3", "ens4_adv_inc_v3", "ens_adv_inc_res_v2"]:
        target_model = get_model(args.target_model,args.model_dir) 
    else:
        target_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    timm.create_model(args.target_model, pretrained=True))
    source_model, target_model = source_model.to(device).eval(), target_model.to(device).eval()
    # process images
    acc=test(test_loader,device,args.epsilon,source_model,target_model,args.attack_name,args.iter_num,args.beta,args.decay_factor,args.Tpt,args.N,args.S,args.n,args.lr,args.N_vt,args.beta_vt)

    with open('log.txt', 'a') as f:
        f.write('beta={},atk={},N={},S={},n={},iter_num={}\n,acc={:.3f},{}-->{}\n'.format(args.beta,args.attack_name,args.N,args.S,args.n,args.iter_num,acc,args.source_model,args.target_model))
    
if __name__ == '__main__':
    main()


