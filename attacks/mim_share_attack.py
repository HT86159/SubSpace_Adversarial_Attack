import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils import *
import pdb
#mim_attack_s
#mim_share_attack
def mim_attack_s(ori_image, eps, iter_num, model, criterion, mu, target, device, momentum):
    if iter_num==0:
        return ori_image,torch.zeros_like(ori_image).detach().to(device)
    alpha = eps / iter_num
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    # momentum = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()

    for _ in range(iter_num):
        adv_image.requires_grad = True
        output = model(adv_image)[0].reshape(1,-1)
        loss = criterion(output,target)

        # model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad + momentum * mu
        momentum = grad

        adv_image = adv_image.detach() + alpha * momentum.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image, momentum



def mim_share_attack(ori_image, eps, iter_num, model, criterion, mu, target, device, S, N, Tpt):
    momentum = torch.zeros_like(ori_image).detach().to(device)
    minx = ori_image - eps
    maxx = ori_image + eps
    image = ori_image.clone().detach().to(device)
    image, momentum = mim_attack_s(image, eps, S, model, criterion, mu, target, device, momentum)
    adv_image = torch.zeros_like(image).to(device)
    for _ in range(N):
        delta = torch.randn_like(image) / Tpt
        perturbed_image = image + delta
        perturbed_image = torch.clamp(perturbed_image,min=minx,max=maxx)
        perturbed_image = torch.clamp(perturbed_image,min=0,max=1)
        image,temp = mim_attack(perturbed_image, eps, iter_num - S, model, criterion, mu, target, device, momentum)
        adv_image = adv_image + image / N
    adv_image = torch.clamp(adv_image,min=minx,max=maxx)
    adv_image = torch.clamp(adv_image,min=0,max=1)
    return adv_image









# def mim_share_attack(ori_image, eps, iter_num, model, criterion, mu, target, device, S, N):
#     minx = ori_image - eps
#     maxx = ori_image + eps
#     image = ori_image.clone().detach().to(device)
#     image = mim_attack(image, eps, S, model, criterion, mu, target, device)
#     adv_image = torch.zeros_like(image).to(device)
#     for _ in range(N):
#         delta = torch.randn_like(image) / 1000
#         image = mim_attack(image + delta , eps, iter_num - S, model, criterion, mu, target, device)
#         adv_image = adv_image + image / N
#     adv_image = torch.clamp(adv_image,min=minx,max=maxx)
#     adv_image = torch.clamp(adv_image,min=0,max=1)
#     return adv_image
