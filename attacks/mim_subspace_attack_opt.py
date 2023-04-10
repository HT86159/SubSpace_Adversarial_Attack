import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils import *
import pdb
from random import sample
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
"""
##sgd
def mim_simplex_opt_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N, lr):
    # pdb.set_trace()

    # 设置基本参数 
    maxx = torch.min(ori_image + eps,torch.tensor(1))
    minx = torch.max(ori_image - eps,torch.tensor(0))

    # 设置变量至gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    # 设置参数与初始化
    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.randn(tuple(image_shape)) / 100
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).detach()#torch.Size([N, 1, 3, 299, 299])
    ADV_image = torch.max(torch.min(ADV_image, torch.tensor(1)), torch.tensor(0)).clone().detach()

    W = to_latent_space(ADV_image, minx, maxx)
    # w = inverse_tanh_space(image).detach()
    l = range(N)
    # print(ADV_image.shape)
    softmax_0 = nn.Softmax(dim=0)
    optimizer = torch.optim.Adam([W], lr=lr)
    for t in range(iter_num):
        # pdb.set_trace()
        # 集成扰动
        # print(W[0][0][0][1])
        W.requires_grad = True
        ADV_image = to_image_space(W, minx, maxx)
        gamma = torch.rand(N, 1, 1, 1, 1).to(device)
        gamma = gamma / gamma.sum(0).item()
        adv_image = (gamma * ADV_image).sum(0)
        
        logits = model(adv_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        CE = criterion(log_softmax.reshape(1,-1),target)#0
        j, k = sample(l,2) 
        adv_image1 = ADV_image[j]
        adv_image2 = ADV_image[k]
        cosim = cosine_similarity(adv_image1, adv_image2)**2
        loss = -CE + beta * cosim
        print('loss={:.4f}\tCE={:.4f}\tcosim={:.4f}'.format(loss.item(),CE.item(),cosim.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # W = torch.max(torch.min(W, maxw), minw).clone().detach()
        # W = inverse_tanh_space(ADV_image)
        # W = inverse_tanh_space(ADV_image).detach()
    ADV_image = to_image_space(W, minx, maxx)
    ADV_image = torch.clamp(ADV_image, min=0, max=1).detach()
    return ADV_image.sum(0) / N
