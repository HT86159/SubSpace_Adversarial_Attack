import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils import *
import pdb
from random import sample,choice
import math
#mim_attack
#mim_rmomt_attack
#mim_subspace_attack
#
def mim_attack(ori_image, eps, iter_num, model, criterion, mu, target, device):
    # pdb.set_trace() 
    alpha = eps / iter_num
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    softmax = nn.Softmax(dim=0)

    momentum = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()
    # pdb.set_trace()
    for _ in range(iter_num):
        adv_image.requires_grad = True
        logits = model(adv_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        loss = criterion(log_softmax.reshape(1,-1),target)#0

        # model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)#!!
        grad = grad + momentum * mu
        momentum = grad

        adv_image.data = adv_image.data + alpha * momentum.sign()
        adv_image.grad.zero_()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image
def mim_line_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device):
    # 设置基本参数 
    alpha = eps / iter_num
    maxx = ori_image + eps
    minx = ori_image - eps
    # 设置变量至gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    # 设置参数
    momentum1 = torch.zeros_like(image).detach().to(device)
    momentum2 = torch.zeros_like(image).detach().to(device)
    # 初始化
    delta1 = torch.randn_like(image) / 100
    delta2 = torch.zeros_like(image)
    adv_image1 = (image.clone() + delta1).detach()#image.clone().detach()
    adv_image2 = (image.clone() + delta2).detach()#image.clone().detach()
    softmax = nn.Softmax(dim=0)


    for t in range(iter_num):
        # 集成扰动
        # pdb.set_trace()

        adv_image1.requires_grad = True
        adv_image2.requires_grad = True
        gamma = torch.rand(1).to(device)
        adv_image = gamma*adv_image1 + (1-gamma)*adv_image2
        
        logits = model(adv_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        CE = criterion(log_softmax.reshape(1,-1),target)#0
        loss = CE - beta * cosine_similarity(delta1, delta2)**2
        # print('loss={:.3f}\tCE={:3f}\tcos={:3f}'.format(loss.item(),CE.item(),(CE.item() - loss.item())))
        loss.backward()
        grad1 = adv_image1.grad.data
        grad1 = grad1 / torch.mean(torch.abs(grad1), dim=(1,2,3), keepdim=True)#!!
        grad1 = grad1 + momentum1 * mu
        momentum1 = grad1
        
        grad2 = adv_image2.grad.data
        grad2 = grad2 / torch.mean(torch.abs(grad2), dim=(1,2,3), keepdim=True)#!!
        grad2 = grad2 + momentum2 * mu
        momentum2 = grad2

        adv_image1.data = adv_image1.data + alpha * momentum1.sign()
        adv_image2.data = adv_image2.data + alpha * momentum2.sign()
        adv_image1.grad.zero_()
        adv_image2.grad.zero_()
        delta1 = torch.clamp(adv_image1 - image, min=-eps, max=eps)
        adv_image1 = torch.clamp(image + delta1, min=0, max=1).detach()
        delta2 = torch.clamp(adv_image2 - image, min=-eps, max=eps)
        adv_image2 = torch.clamp(image + delta2, min=0, max=1).detach()
    return (adv_image1 + adv_image2) / 2
def mim_simplex_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N):
    alpha = eps / iter_num
    # 设置变量至gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    # 设置参数与初始化
    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.zeros(tuple(image_shape)) / 100
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).detach()#torch.Size([N, 1, 3, 299, 299])
    Momentum = torch.zeros_like(Delta).detach().to(device)
    ORI_image = (image.clone() + Momentum).detach()
    l = range(N)
    # print(ADV_image.shape)
    softmax_0 = nn.Softmax(dim=0)
    for t in range(iter_num):
    # 集成扰动
    # pdb.set_trace()
        ADV_image.requires_grad = True
        gamma = torch.rand(N, 1, 1, 1, 1).to(device)
        gamma = gamma / gamma.sum(0).item()
        adv_image = (gamma * ADV_image).sum(0)
        
        logits = model(adv_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        CE = criterion(log_softmax.reshape(1,-1),target)#0
        cosim = 0
        for _ in range(N*(N-1)):
            j, k = sample(l,2) 
            adv_image1 = ADV_image[j]
            adv_image2 = ADV_image[k]
            cosim += cosine_similarity(adv_image1, adv_image2)**2
        loss = CE - beta * cosim / (N*(N-1))
        # print('loss={:.4f}\tCE={:.4f}\tcosim={:.4f}'.format(loss.item(),CE.item(),cosim.item()))
        loss.backward()
        Grad = ADV_image.grad.data
        Grad = Grad / torch.mean(torch.abs(Grad), dim=(1,2,3,4), keepdim=True)#!!
        Grad = Grad + Momentum * mu
        Momentum = Grad
        
        ADV_image.data = ADV_image.data + alpha * Momentum.sign()
        ADV_image.grad.zero_()
        Delta = torch.clamp(ADV_image - ORI_image, min=-eps, max=eps)
        ADV_image = torch.clamp(ORI_image + Delta, min=0, max=1).detach()
    return ADV_image.sum(0) / N
# def mim_simplex_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N):
    alpha = eps / iter_num
    # 设置变量至gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    # 设置参数与初始化
    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.zeros(tuple(image_shape)) / 100
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).detach()#torch.Size([N, 1, 3, 299, 299])
    Momentum = torch.zeros_like(Delta).detach().to(device)
    ORI_image = (image.clone() + Momentum).detach()
    l = range(N)
    # print(ADV_image.shape)
    softmax_0 = nn.Softmax(dim=0)
    for t in range(iter_num):
    # 集成扰动
    # pdb.set_trace()
        ADV_image.requires_grad = True
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
        loss = CE - beta * cosim
        print('loss={:.4f}\tCE={:.4f}\tcosim={:.4f}'.format(loss.item(),CE.item(),cosim.item()))
        loss.backward()
        Grad = ADV_image.grad.data
        Grad = Grad / torch.mean(torch.abs(Grad), dim=(1,2,3,4), keepdim=True)#!!
        Grad = Grad + Momentum * mu
        Momentum = Grad
        
        ADV_image.data = ADV_image.data + alpha * Momentum.sign()
        ADV_image.grad.zero_()
        Delta = torch.clamp(ADV_image - ORI_image, min=-eps, max=eps)
        ADV_image = torch.clamp(ORI_image + Delta, min=0, max=1).detach()
    return ADV_image.sum(0) / N


# def vmim_simplex_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N):
    alpha = eps/iter_num
    N_vt=20
    beta_vt=3/2
    #设置变量在gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)

    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.randn(tuple(image_shape)) / 100
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).clone().detach()#torch.Size([N, 1, 3, 299, 299])
    Momentum = torch.zeros_like(Delta).detach().to(device)
    ORI_image = (image.clone() + Momentum).detach()
    V = torch.zeros_like(image).detach().to(device)
    # 设置参数与初始化
    l = range(N)
    softmax_0 = nn.Softmax(dim=0)
    for t in range(iter_num):
    # pdb.set_trace()
        ADV_image.requires_grad = True
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
        cosim = cosine_similarity(adv_image1-image, adv_image2-image)**2
        loss = CE - beta * cosim
        # print('loss={:.4f}\tCE={:.4f}\tcosim={:.4f}'.format(loss.item(),CE.item(),cosim.item()))
        loss.backward()

        ADV_Grad = ADV_image.grad.data
        Grad = (ADV_Grad + V) / torch.mean(torch.abs(ADV_Grad+ V), dim=(1,2,3,4), keepdim=True)#!!
        Grad = Grad + Momentum * mu
        Momentum = Grad

        GV_grad = torch.zeros_like(Delta).detach().to(device)
        for i in range(N_vt):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(adv_image).uniform_(-eps * beta_vt, eps * beta_vt)
            neighbor_images.requires_grad = True
            outputs = model(neighbor_images)[0]
            softmaxs = softmax_0(outputs)
            log_softmaxs = torch.log(softmaxs)
            # print(softmax)
            neighbor_loss = criterion(log_softmaxs.reshape(1,-1),target)
            neighbor_loss.backward()

            neighbor_grad = neighbor_images.grad.data
            GV_grad += neighbor_grad

        V = GV_grad / N_vt - ADV_Grad

        ADV_image.data = ADV_image.data + alpha * Momentum.sign()
        ADV_image.grad.zero_()
        Delta = torch.clamp(ADV_image - ORI_image, min=-eps, max=eps)
        ADV_image = torch.clamp(ORI_image + Delta, min=0, max=1).detach()
    return ADV_image.sum(0) / N

##the reason:zeros\betavt
##findreason
# def vmim_simplex_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N, N_vt, beta_vt):
    alpha = eps/iter_num
    #设置变量在gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)

    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.rand(tuple(image_shape)) / 100
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).clone().detach()#torch.Size([N, 1, 3, 299, 299])
    Momentum = torch.zeros_like(Delta).detach().to(device)
    ORI_image = (image.clone() + Momentum).detach()
    V = torch.zeros_like(ADV_image).detach().to(device)
    # 设置参数与初始化
    l = range(N)
    softmax_0 = nn.Softmax(dim=0)
    for t in range(iter_num):
    # pdb.set_trace()
        ADV_image.requires_grad = True
        gamma = torch.rand(N, 1, 1, 1, 1).to(device)
        gamma = gamma / gamma.sum(0).item()
        adv_image = (gamma * ADV_image).sum(0)
        
        logits = model(adv_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        CE = criterion(log_softmax.reshape(1,-1),target)
        j, k = sample(l,2) 
        adv_image1 = ADV_image[j]
        adv_image2 = ADV_image[k]
        cosim = cosine_similarity(adv_image1-image, adv_image2-image)**2
        loss = CE - beta * cosim
        # print('loss={:.4f}\tCE={:.4f}\tcosim={:.4f}'.format(loss.item(),CE.item(),cosim.item()))
        loss.backward()

        ADV_Grad = ADV_image.grad.data
        Grad = (ADV_Grad + V) / torch.mean(torch.abs(ADV_Grad + V), dim=(1,2,3,4), keepdim=True)
        Grad = Grad + Momentum * mu
        Momentum = Grad

        GV_grad = torch.zeros_like(image).detach().to(device)
        for i in range(N_vt):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(adv_image).uniform_(-eps * beta_vt, eps * beta_vt)
            neighbor_images.requires_grad = True
            outputs = model(neighbor_images)[0]
            softmaxs = softmax_0(outputs)
            log_softmaxs = torch.log(softmaxs)
            # print(softmax)
            neighbor_loss = criterion(log_softmaxs.reshape(1,-1),target)
            neighbor_loss.backward()

            neighbor_grad = neighbor_images.grad.data
            GV_grad += neighbor_grad

        V = GV_grad / N_vt - ADV_Grad

        ADV_image.data = ADV_image.data + alpha * Momentum.sign()
        ADV_image.grad.zero_()
        Delta = torch.clamp(ADV_image - ORI_image, min=-eps, max=eps)
        ADV_image = torch.clamp(ORI_image + Delta, min=0, max=1).detach()
    return ADV_image.sum(0) / N



def vmim_simplex_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N, N_vt,beta_vt):
    alpha = eps/iter_num
    #设置变量在gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)

    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.zeros(tuple(image_shape)) 
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).clone().detach()#torch.Size([N, 1, 3, 299, 299])
    Momentum = torch.zeros_like(Delta).detach().to(device)
    ORI_image = (image.clone() + Momentum).detach()
    V = torch.zeros_like(image).detach().to(device)
    # 设置参数与初始化
    l = range(N)
    softmax_0 = nn.Softmax(dim=0)
    for t in range(iter_num):
        # pdb.set_trace()
        ADV_image.requires_grad = True
        gamma = torch.rand(N, 1, 1, 1, 1).to(device)
        gamma = gamma / gamma.sum(0).item()
        adv_image = (gamma * ADV_image).sum(0)
        delta =  torch.clamp(adv_image - ori_image, min=-eps, max=eps)
        adv_image = torch.clamp(adv_image + delta, min=0, max=1)
        
        logits = model(adv_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        CE = criterion(log_softmax.reshape(1,-1),target)#0

        j, k = sample(l,2) 
        adv_image1 = ADV_image[j]
        adv_image2 = ADV_image[k]
        cosim = cosine_similarity(adv_image1-image, adv_image2-image)**2
        loss = CE - beta * cosim

        loss.backward()

        ADV_Grad = ADV_image.grad.data#[10 1 3 299 299]
        Grad = (ADV_Grad + V) / torch.mean(torch.abs(ADV_Grad+ V), dim=(1,2,3,4), keepdim=True)#!!
        Grad = Grad + Momentum * mu
        Momentum = Grad

        GV_grad = torch.zeros_like(Delta).detach().to(device)
        for _ in range(N_vt):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(adv_image).uniform_(-eps * beta_vt, eps * beta_vt)
            neighbor_images.requires_grad = True
            outputs = model(neighbor_images)[0]
            softmaxs = softmax_0(outputs)
            log_softmaxs = torch.log(softmaxs)
            # print(softmax)
            neighbor_loss = criterion(log_softmaxs.reshape(1,-1),target)
            neighbor_loss.backward()

            neighbor_grad = neighbor_images.grad.data
            GV_grad += neighbor_grad
        ij = sample(l,1)
        V = GV_grad / N_vt - ADV_Grad

        ADV_image.data = ADV_image.data + alpha * Momentum.sign()
        ADV_image.grad.zero_()
        ADV_image = ADV_image.clone().detach()
        if t != iter_num-1:
            Delta = torch.clamp(ADV_image - ORI_image, min=-eps, max=eps)
            ADV_image = torch.clamp(ORI_image + Delta, min=0, max=1).detach()
        else:
            output_image = ADV_image.sum(0) / N
            output_delta = torch.clamp(output_image - ori_image, min=-eps, max=eps)
            output_image = torch.clamp(ori_image + output_delta, min=0, max=1).detach()
    return output_image

def vnim_simplex_attack(ori_image, eps, iter_num, model, mu, target, beta, criterion, device ,N, N_vt,beta_vt):
    alpha = eps/iter_num
    #设置变量在gpu
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)

    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.zeros(tuple(image_shape)) 
    Delta = Delta.to(device)#torch.Size([N, 1, 3, 299, 299])
    ADV_image = (image.clone() + Delta).clone().detach()#torch.Size([N, 1, 3, 299, 299])
    Momentum = torch.zeros_like(Delta).detach().to(device)
    ORI_image = (image.clone() + Momentum).detach()
    V = torch.zeros_like(image).detach().to(device)
    # 设置参数与初始化
    l = range(N)
    softmax_0 = nn.Softmax(dim=0)
    for t in range(iter_num):
        # pdb.set_trace()
        ADV_image.requires_grad = True
        NES_image = ADV_image + mu * alpha * Momentum
        gamma = torch.rand(N, 1, 1, 1, 1).to(device)
        gamma = gamma / gamma.sum(0).item()
        adv_image = (gamma * ADV_image).sum(0)
        delta =  torch.clamp(adv_image - ori_image, min=-eps, max=eps)
        adv_image = torch.clamp(adv_image + delta, min=0, max=1)
        nes_image = (gamma * NES_image).sum(0)
        nes_delta =  torch.clamp(nes_image - ori_image, min=-eps, max=eps)
        nes_image = torch.clamp(nes_image + nes_delta, min=0, max=1)
        
        logits = model(nes_image)[0]#正常
        softmax = softmax_0(logits)
        log_softmax = torch.log(softmax)
        # print(softmax)
        CE = criterion(log_softmax.reshape(1,-1),target)#0

        j, k = sample(l,2) 
        adv_image1 = ADV_image[j]
        adv_image2 = ADV_image[k]
        cosim = cosine_similarity(adv_image1-image, adv_image2-image)**2
        loss = CE - beta * cosim

        loss.backward()

        ADV_Grad = ADV_image.grad.data#[10 1 3 299 299]
        Grad = (ADV_Grad + V) / torch.mean(torch.abs(ADV_Grad+ V), dim=(1,2,3,4), keepdim=True)#!!
        Grad = Grad + Momentum * mu
        Momentum = Grad

        GV_grad = torch.zeros_like(Delta).detach().to(device)
        for _ in range(N_vt):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(adv_image).uniform_(-eps * beta_vt, eps * beta_vt)
            neighbor_images.requires_grad = True
            outputs = model(neighbor_images)[0]
            softmaxs = softmax_0(outputs)
            log_softmaxs = torch.log(softmaxs)
            # print(softmax)
            neighbor_loss = criterion(log_softmaxs.reshape(1,-1),target)
            neighbor_loss.backward()

            neighbor_grad = neighbor_images.grad.data
            GV_grad += neighbor_grad
        V = GV_grad / N_vt - ADV_Grad

        ADV_image.data = ADV_image.data + alpha * Momentum.sign()
        ADV_image.grad.zero_()
        ADV_image = ADV_image.clone().detach()
        if t != iter_num-1:
            Delta = torch.clamp(ADV_image - ORI_image, min=-eps, max=eps)
            ADV_image = torch.clamp(ORI_image + Delta, min=0, max=1).detach()
        else:
            output_image = ADV_image.sum(0) / N
            output_delta = torch.clamp(output_image - ori_image, min=-eps, max=eps)
            output_image = torch.clamp(ori_image + output_delta, min=0, max=1).detach()
    return output_image
