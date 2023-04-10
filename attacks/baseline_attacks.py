import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils import *
import pdb
#mim_attack
#bim_attack
#nim_attack
#vmim_attack
#vnim_attack
#mim_ens_attack
def mim_attack(ori_image, eps, iter_num, model, criterion, mu, target, device):
    alpha = eps / iter_num
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)

    momentum = torch.zeros_like(image).detach().to(device)
    adv_image = ori_image.clone().detach().to(device)
    loss_list=[]
    for _ in range(iter_num):
        adv_image.requires_grad = True
        output = model(adv_image)[0].reshape(1,-1)
        loss = criterion(output,target)
        loss_list.append(loss.item())
        # model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad + momentum * mu
        momentum = grad

        adv_image = adv_image.detach() + alpha * momentum.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    # print(loss_list)
    return adv_image
def bim_attack(image,eps,iter_num,model,criterion,target,device):
    alpha = eps/iter_num
    image = image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    ori_image = image.clone().detach()

    for iter in range(iter_num):
        image.requires_grad = True
        output = model(image)[0].reshape(1,-1)
        loss = criterion(output,target)
        model.zero_grad()
        loss.backward()

        image_grad = image.grad.data
        grad_sign = image_grad.sign()
        adv_image = image + alpha*grad_sign
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        image = torch.clamp(image + delta, min=0, max=1).detach()
    return image
def nim_attack(image, eps, iter_num, model, criterion, mu, target, device):
    alpha = eps/iter_num
    g = 0
    image = image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    momentum = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()

    for t in range(iter_num):
        adv_image.requires_grad = True
        nes_images = adv_image + mu * alpha *momentum
        output = model(nes_images)
        loss = criterion(output,target)
        model.zero_grad()
        loss.backward()

        grad = adv_image.grad.data
        grad = momentum * mu + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad

        adv_image = adv_image.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image
def vmim_attack(image, eps, iter_num, model, criterion, mu, target, device, N_vt, beta_vt):
    alpha = eps/iter_num
    image = image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    momentum = torch.zeros_like(image).detach().to(device)
    v = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()

    for t in range(iter_num):
        adv_image.requires_grad = True
        output = model(adv_image)[0].reshape(1,-1)
        loss = criterion(output,target)
        loss.backward()

        adv_grad = adv_image.grad.data
        grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
        grad = momentum * mu + grad 
        momentum = grad

        GV_grad = torch.zeros_like(image).detach().to(device)
        for i in range(N_vt):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(image).uniform_(-eps * beta_vt, eps * beta_vt)
            neighbor_images.requires_grad = True
            outputs = model(neighbor_images)[0].reshape(1,-1)

            neighbor_loss = criterion(outputs,target)
            neighbor_loss.backward()

            neighbor_grad = neighbor_images.grad.data
            GV_grad += neighbor_grad

        v = GV_grad / N_vt - adv_grad

        adv_image = adv_image.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image

def vnim_attack(image, eps, iter_num, model, criterion, mu, target, device):
    alpha = eps/iter_num
    N=20
    beta=3/2
    grad = 0
    image = image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    momentum = torch.zeros_like(image).detach().to(device)
    v = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()

    for t in range(iter_num):
        adv_image.requires_grad = True
        nes_image = adv_image + mu * alpha * momentum
        output = model(nes_image)[0].reshape(1,-1)
        loss = criterion(output,target)
        model.zero_grad()
        loss.backward()

        adv_grad = adv_image.grad.data
        grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
        grad = momentum * mu + grad 
        momentum = grad

        GV_grad = torch.zeros_like(image).detach().to(device)
        for i in range(N):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(image).uniform_(-eps * beta, eps * beta)
            neighbor_images.requires_grad = True
            outputs = model(neighbor_images)[0].reshape(1,-1)

            neighbor_loss = criterion(outputs,target)
            model.zero_grad()
            neighbor_loss.backward()

            neighbor_grad = neighbor_images.grad.data
            GV_grad += neighbor_grad

        v = GV_grad / N - adv_grad

        adv_image = adv_image.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image


def mim_ens_attack(ori_image, eps, T, model, criterion, y_ture, beta, mu, device, Tpt, N):##生成两个clip之后的delta然后集成再clip
    adv_image = torch.zeros_like(ori_image)
    maxx = ori_image + eps
    minx = ori_image - eps
    for n in range(N):
        image = ori_image.clone().detach()
        mim_image = mim_attack(image, eps, T, model, criterion, mu, y_ture, device)
        adv_image.data = adv_image.data + mim_image.data / N
    delta = torch.clamp(adv_image - ori_image , min=-eps, max=eps)
    adv_image = torch.clamp(ori_image + delta, min=0, max=1).detach()
    return adv_image


def vmim_ens_attack(ori_image, eps, iter_num, model, criterion, target, mu, device, N, N_vt, beta_vt):##生成两个clip之后的delta然后集成再clip
    adv_image = torch.zeros_like(ori_image)
    for _ in range(N):
        image = ori_image.clone().detach()
        vmim_image = vmim_attack(image, eps, iter_num, model, criterion, mu, target, device, N_vt, beta_vt)
        adv_image.data = adv_image.data + vmim_image.data / N
    delta = torch.clamp(adv_image - ori_image , min=-eps, max=eps)
    adv_image = torch.clamp(ori_image + delta, min=0, max=1).detach()
    return adv_image
