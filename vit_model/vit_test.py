# coding=utf-8
import argparse
import os.path
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import *

parser = argparse.ArgumentParser(description='eval attack in PyTorch')

parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--input_csv', default='./data/input_dir/dev.csv', type=str, help='csv info of clean examples')
parser.add_argument('--input_dir', default='./data/input_dir/images/', type=str, help='directory of clean examples')
parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import timm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from utils.loader import NIPS_GAME
from utils.attack_method import save_img

def entropy(px):
    return - torch.sum(F.softmax(px, dim=1) * F.log_softmax(px, dim=1), dim=1)
    
def main():
    # todo: parse input_dir and get output log file
    components = os.path.abspath(args.input_dir).split('/')
    if 'adv' in components:
        idx = components.index('adv')
        log_file = os.path.join("results/ijcai/vit", "/".join(components[idx + 1:])) + ".log"
        data_csv_file = os.path.join("results/ijcai/vit", "/".join(components[idx + 1:])) + "_data.csv"
        etp_csv_file = os.path.join("results/ijcai/vit", "/".join(components[idx + 1:])) + "_etp.csv"
    else:
        log_file = os.path.join("results/ijcai/vit", "images.log")
        data_csv_file = os.path.join("results/ijcai/vit", "images_data.csv")
        etp_csv_file = os.path.join("results/ijcai/vit", "images_etp.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # model_names = ["resnet50", "densenet121", "resnet101", "vgg19", "densenet169", "inception_v3"]
    # model_names = ["vgg19"]
    # model_names = ['vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_base_patch16_224', 'vit_base_patch16_224_in21k', 'vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_base_patch32_224', 'vit_base_patch32_224_in21k', 'vit_small_patch16_224', 'vit_small_patch16_224_in21k', 'vit_small_patch32_224', 'vit_small_patch32_224_in21k', 'deit_base_patch16_224', 'deit3_base_patch16_224', 'resnetv2_101x3_bitm', 'resnetv2_101x3_bitm_in21k']
    model_names = ['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224', 'deit3_base_patch16_224', 'resnetv2_101x3_bitm']
    res_dict = {model:{"top1":0, "top5":0} for model in model_names}
    model_dict = {model:torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                timm.create_model(model, pretrained=True).eval()).cuda() for model in model_names}
    # model_dict = {model:torch.nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                             getattr(models, model)(pretrained=True).eval()).cuda() for model in model_names}
    datas = np.empty((0, len(model_dict)))
    datas_entropy = np.empty((0, len(model_dict)))
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        for i, (x, name, y, target) in enumerate(clean_loader):
            if args.use_gpu:
                x = x.cuda()
                y = y.cuda()
            results = []
            results_etp = []
            for model_name, model in model_dict.items():
                output = model(x)
                pred_top1 = output.topk(k=1, largest=True).indices
                pred_top5 = output.topk(k=5, largest=True).indices
                if pred_top1.dim() >= 2:
                    pred_top1 = pred_top1.squeeze()

                corr_1 = (pred_top1 == y).sum().item()
                corr_5 = (pred_top5 == y.unsqueeze(dim=1).expand(-1, 5)).sum().item()

                res_dict[model_name]['top1'] += corr_1
                res_dict[model_name]['top5'] += corr_5

                acc = (pred_top1 == y).detach().cpu().float().numpy().reshape(-1, 1)
                out_etp = entropy(output).detach().cpu().float().numpy().reshape(-1, 1)
                results.append(acc)
                results_etp.append(out_etp)
            
            results = np.concatenate(results, axis=1)
            results_etp = np.concatenate(results_etp, axis=1)
            datas = np.concatenate([datas, results], axis=0)
            datas_entropy = np.concatenate([datas_entropy, results_etp], axis=0)
    
    # print(args.input_dir)
    # print(log_file)
    for model_name, res in res_dict.items():
        print('%s: top1=%.3f, top5=%.3f' % (model_name, res['top1'] / no_samples, res['top5'] / no_samples))

    original_stdout = sys.stdout
    with open(log_file, 'a') as f:
        sys.stdout = f
        for model_name, res in res_dict.items():
            print('%s: top1=%.3f, top5=%.3f' % (model_name, res['top1'] / no_samples, res['top5'] / no_samples))
        sys.stdout = original_stdout
    
    df_data = pd.DataFrame(datas, columns=list(model_dict.keys()))
    df_data.to_csv(data_csv_file, index=0)
    
    df_etp = pd.DataFrame(datas_entropy, columns=list(model_dict.keys()))
    df_etp.to_csv(etp_csv_file, index=0)

if __name__ == '__main__':
    # data_loader
    preprocess = transforms.Compose([
    transforms.ToTensor()])
    clean_dataset = NIPS_GAME(args.input_dir, args.input_csv, preprocess)
    test_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False,  pin_memory=True)
    # use cuda
    device=torch.device('cuda:{}'.format(str(args.device % 8)) if args.use_cuda and torch.cuda.is_available else 'cpu')
    # load pretrain_models
    source_model, target_model = get_model(args.source_model,args.target_model,args.model_dir)
    source_model = source_model.to(device).eval()
    target_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                timm.create_model('vit_base_patch16_244', pretrained=True).eval()).to(device))

    # model_names = ['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224', 'deit3_base_patch16_224', 'resnetv2_101x3_bitm']
    main()
