device=3
M_list=(10)
# 2 2 2 2)
lr_list=(0.000005)
iter_num_list=(20 100 300)
#9 10)
N_list=(10)
beta_list=(0)
N_vt_list=(0)
beta_vt_list=(1.5)
Tpt_list=(100)
loss=("logits")
attack_name_list=("ct_vmim")
#'mim_sub_line')
# source_model_list=("inception_v3")
# source_model_list=('resnet50' "inception_v3" "vgg16" "dense121")
source_model_list=('resnet50')
# source_model_list=('resnet_v2_152')

# 'inception_v3' "resnet_v2_50"  'resnet_v2_101' 'resnet_v2_152'
# "inc_res_v2" "inception_v4" "vgg16" "dense121"
# "adv_inception_v3" "ens3_adv_inc_v3" "ens4_adv_inc_v3" "ens_adv_inc_res_v2"
# "vit_small_patch16_224" "vit_small_patch32_224" "vit_base_patch16_224" "vit_base_patch32_224"
# 'vit_large_patch16_224' "deit_base_patch16_224" "deit3_base_patch16_224" "resnetv2_101x3_bitm"
# 'inception_v3,resnet_v2_50,resnet_v2_152,inc_res_v2'
# target_model_list=("resnet_v2_101,ens3_adv_inc_v3,ens4_adv_inc_v3,ens_adv_inc_res_v2")
target_model_list=("inception_v3,resnet50,vgg16,dense121")
# resnet_v2_101,inception_v3,inception_v4,inc_res_v2,ens3_adv_inc_v3,ens4_adv_inc_v3,ens_adv_inc_res_v2
# "vit_small_patch16_224,vit_small_patch32_224,vit_base_patch16_224,vit_base_patch32_224,vit_large_patch16_224,deit_base_patch16_224,deit3_base_patch16_224"
# "inception_v3,resnet_v2_50,resnet_v2_101,resnet_v2_152,inc_res_v2,inception_v4,vgg16,dense121"
# "adv_inception_v3,ens3_adv_inc_v3,ens4_adv_inc_v3,ens_adv_inc_res_v2"
for source_model in ${source_model_list[@]}
do
    for target_model in ${target_model_list[@]}
    do
        for attack_name in ${attack_name_list[@]}
        do
            for M in ${M_list[@]}
            do
                for N in ${N_list[@]}
                do
                    for iter_num in ${iter_num_list[@]}
                    do
                        for lr in ${lr_list[@]}
                        do
                            for N_vt in ${N_vt_list[@]}
                            do
                                for beta_vt in ${beta_vt_list[@]}
                                do
                                    for beta in ${beta_list[@]}
                                    do
                                        for Tpt in ${Tpt_list[@]}
                                        do
                                            device=$(($device+1))
                                            python /data/huangtao/projects/gradient_attack/subsapce-attack/eval/eval_cnn_target.py \
                                            --attack_name $attack_name \
                                            --M $M \
                                            --source_model $source_model \
                                            --target_model $target_model \
                                            --iter_num $iter_num \
                                            --device $device \
                                            --N $N \
                                            --beta $beta \
                                            --iter_num $iter_num \
                                            --lr $lr \
                                            --N_vt $N_vt \
                                            --Tpt $Tpt \
                                            --loss $loss \
                                            --beta_vt $beta_vt &
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done