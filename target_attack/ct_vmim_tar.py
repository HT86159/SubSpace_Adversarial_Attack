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
# diversity_prob = 0.5


class Translation_Kernel:
    def __init__(self, len_kernel=15, nsig=3, kernel_name='gaussian'):
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

def scale_transform(input_tensor, m=1):
    # outs = [(input_tensor) / (2**i) for i in range(m)]
    outs = [(input_tensor)  for i in range(m)]
    x_batch = torch.cat(outs, dim=0)
    return x_batch

def input_diversity(input_tensor, resize=330, diversity_prob=0.5):
    if torch.rand(1) >= diversity_prob:
        return input_tensor
    image_width = input_tensor.shape[-1]
    assert image_width == 299, "only support ImageNet"
    rnd = torch.randint(image_width, resize, ())
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = resize - rnd
    w_rem = resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    # padded = nn.functional.interpolate(padded, (image_width, image_width), mode='bilinear', align_corners=False)
    return padded
def DI(X_in):
    rnd = np.random.randint(299, 330,size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out
    else:
        return  X_in



def vmim_ct_target_attack(image, eps, iter_num, model_list, criterion, mu, target, device, N_vt, beta_vt, gaussian_kernel, is_tf_model,loss_f):
    alpha = 2 / 255
    image = image.clone().detach().to(device)
    target = target.clone().detach().to(device)
    momentum = torch.zeros_like(image).detach().to(device)
    v = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()
    m = 1
    for t in range(iter_num):

        # pdb.set_trace()
        adv_image.requires_grad = True
        # pdb.set_trace()
        si_adv_image = scale_transform(adv_image, m=m)
        # di_adv_image = input_diversity(si_adv_image, resize=330, diversity_prob=0.7)
        di_adv_image = DI(si_adv_image)
        logits = 0
        # pdb.set_trace()
        for model in model_list:
            if is_tf_model:
                logits += model(di_adv_image)[0].reshape(m,-1)/len(model_list)
            else:
                logits += model(di_adv_image).reshape(m,-1) / len(model_list)
        targets = target.repeat(m)
        if loss_f == "ce":
            loss = F.cross_entropy(logits, targets, reduction="sum")
        elif loss_f == "logits":
            predicted_logits = logits[torch.arange(len(targets)), targets]
            loss = -torch.mean(predicted_logits)
        else:
            raise ValueError("Invalid loss function specified")
        loss.backward()

        # print(loss)
        new_grad = adv_image.grad
            # adv_image = adv_image.detach()

        current_grad = new_grad + v
        grad =  F.conv2d(current_grad, gaussian_kernel, stride=1, padding='same', groups=3)
        # grad =  F.conv2d(current_grad, gaussian_kernel, stride=1, padding=(2,2), groups=3)
        grad = (grad) / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = momentum * mu + grad
        momentum = grad

        gv_grad = torch.zeros_like(image).detach().to(device)
        for i in range(N_vt):
            neighbor_images = adv_image.detach() + \
                                torch.randn_like(image).uniform_(-eps * beta_vt, eps * beta_vt)
            neighbor_images.requires_grad = True
            si_neighbors = scale_transform(neighbor_images, m=m)

            di_neighbors = input_diversity(si_neighbors, resize=330, diversity_prob=0.3)
            outputs_neighbors = 0
            if is_tf_model:
                outputs_neighbors += model(di_neighbors)[0].reshape(m,-1)/len(model_list)
            else:
                outputs_neighbors += model(di_neighbors).reshape(m,-1)/len(model_list)
            targets = target.repeat(m)
            neighbor_loss = F.cross_entropy(outputs_neighbors, targets, reduction="sum")
            neighbor_loss.backward()
            neighbor_grad = neighbor_images.grad.data
            gv_grad += neighbor_grad
        if N_vt == 0:
            v = 0
        else:
            v = gv_grad / N_vt - new_grad
        adv_image.grad.zero_()
        adv_image = adv_image.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
        delta = adv_image - image

    # import pdb;pdb.set_trace()
    return adv_image
