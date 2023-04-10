import sys
sys.path.append('..')
from utils import *
import timm


use_cuda = True
device = 0
source_model_list = ['resnet_v2_50','resnet_v2_101',"resnet_v2_152"]
target_model = 'resnet_v2_50'
model_dir = "/data/huangtao/models/torch_net_weight/"
vit_name = 'vit_base_patch16_224'

preprocess = transforms.Compose([
transforms.ToTensor()])
clean_dataset = NIPS_GAME("/data/public/data/nipsgame2017_dataset", "/data/public/data/nipsgame2017_dataset_py/dev_dataset.csv", preprocess)
test_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False,  pin_memory=True)
# use cuda
device=torch.device('cuda:{}'.format(str(device % 8)) if use_cuda and torch.cuda.is_available else 'cpu')
# load pretrain_models
# model_names = ['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224', 'deit3_base_patch16_224', 'resnetv2_101x3_bitm']
# source_model, target_model = get_model(source_model,'resnet_v2_50',model_dir)
# source_model = source_model.to(device).eval()
# target_model = target_model.to(device).eval()

# source_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             timm.create_model(vit_name, pretrained=True).eval()).to(device)

# import pdb;pdb.set_trace()
vit_list = ['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224', 'deit3_base_patch16_224', 'resnetv2_101x3_bitm']
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
from torchvision import models
source_model=torch.nn.Sequential(normalize,models.vgg16(weights='VGG16_Weights.DEFAULT')).to(device).eval()

for vit_name in vit_list:
    source_model = torch.nn.Sequential(transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 timm.create_model(vit_name, pretrained=True).eval()).to(device)


    iter = 0
    acc= 0
    for data,name,target,TargetClass in test_loader:
        # target = target - 1
        iter+=1
        # print('epoch:{},success:{}'.format(iter,success))
        data,target = data.to(device),target.to(device)
        output = source_model(data)[0]
        pred = torch.argmax(output,axis=0) 
        if pred.item() == target.item()-1:
            acc += 1
        if iter % 1000 == 0:
            print('model={}\tTest complete={}\tacc={:.4f}'.format(vit_name,iter/len(test_loader),acc/iter))


