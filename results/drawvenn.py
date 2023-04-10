from matplotlib import pyplot as plt
from matplotlib_venn import venn2,venn2_circles
import pandas as pd
import torch
import venn
from matplotlib_venn import venn3_circles

# vmim = pd.read_excel('/data/huangtao/projects/subsapce-attack/results/atk=vmim,resnet_v2_50->resnetv2_101x3_bitm,accuracy=0.5190-output.xlsx')
# vmim_s = pd.read_excel('/data/huangtao/projects/subsapce-attack/results/atk=vmim_simplex_attack,resnet_v2_50->resnetv2_101x3_bitm,accuracy=0.5930-output.xlsx')
# vmim_tensor = torch.tensor(vmim[0])
# vmim_s_tensor = torch.tensor(vmim_s[0])
# temp = torch.tensor(range(len(vmim_tensor))) + 1
# vmim_tensor = temp * vmim_tensor
# vmim_s_tensor = temp * vmim_s_tensor
# vmim_list = vmim_tensor.numpy().tolist()
# vmim_s_list = vmim_s_tensor.numpy().tolist()
# vmim_set = set(vmim_list)
# vmim_s_set = set(vmim_s_list) 

# mycolor=[[0.10588235294117647, 0.6196078431372549, 0.4666666666666667,0.6],
#          [0.9058823529411765, 0.1607843137254902, 0.5411764705882353, 0.6]]

# plt.style.use('seaborn-whitegrid')
# # ax.set_axis_on()#开启坐标网格线
# # ax.set_title('venn')
# venn2(subsets = [vmim_set, vmim_s_set,], set_labels=("vmim","vmim_simpex"),set_colors=mycolor)
# plt.title('vmim-vs-vmim_simplex')
# plt.savefig('rn50->rnx3.png')


#验证多攻破的是不是同一些图片
def delta_set(vmim_xlsx,vmim_s_xlsx):
    vmim = pd.read_excel(vmim_xlsx)
    vmim_s = pd.read_excel(vmim_s_xlsx)
    vmim_tensor = torch.tensor(vmim[0])
    vmim_s_tensor = torch.tensor(vmim_s[0])
    temp = torch.tensor(range(len(vmim_tensor))) + 1
    vmim_tensor = temp * vmim_tensor
    vmim_s_tensor = temp * vmim_s_tensor
    vmim_list = vmim_tensor.numpy().tolist()
    vmim_s_list = vmim_s_tensor.numpy().tolist()
    vmim_set = set(vmim_list)
    vmim_s_set = set(vmim_s_list)
    return vmim_s_set - vmim_set
rn50x3_set = delta_set("/data/huangtao/projects/subsapce-attack/results/atk=vmim,resnet_v2_50->resnetv2_101x3_bitm,accuracy=0.5190-output.xlsx", "/data/huangtao/projects/subsapce-attack/results/atk=vmim_simplex_attack,resnet_v2_50->resnetv2_101x3_bitm,accuracy=0.5930-output.xlsx")
deit_set = delta_set("/data/huangtao/projects/subsapce-attack/results/atk=vmim,resnet_v2_50->deit_base_patch16_224,accuracy=0.3140-output.xlsx", "/data/huangtao/projects/subsapce-attack/results/atk=vmim_simplex_attack,resnet_v2_50->deit_base_patch16_224,accuracy=0.3850-output.xlsx")
mycolor=[[0.10588235294117647, 0.6196078431372549, 0.4666666666666667,0.6],
         [0.9058823529411765, 0.1607843137254902, 0.5411764705882353, 0.6]]

plt.style.use('seaborn-whitegrid')
# ax.set_axis_on()#开启坐标网格线
# ax.set_title('venn')
venn2(subsets = [rn50x3_set, deit_set,], set_labels=("rn50x3","deit"),set_colors=mycolor)
plt.title('rn50x3 vs deit')
plt.savefig('rn50 vs rnx3.png')
