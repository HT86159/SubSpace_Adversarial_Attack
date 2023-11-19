device=1
attack_name_list=("ss_vnim" "ss_vmim") 
# "ct_vmim" "ct_vnim" "ss_vmim" "ct_vnim" "ss_ct_vmim" "ss_ct_vnim"

source_model_list=("resnet_v2_50")
# 'inception_v3' "resnet_v2_50"  'resnet_v2_101' 'resnet_v2_152' 
# "inc_res_v2" "vgg19" "dense121" 
# "inception_v3,resnet_v2_50,resnet_v2_152,inc_res_v2"

target_model_list=("inception_v3,resnet_v2_50,resnet_v2_101,resnet_v2_152,inc_res_v2,inception_v4,vgg19,dense121"  )
# 'inception_v3,resnet_v2_50,resnet_v2_101,resnet_v2_152'
# "inc_res_v2" "inception_v4" "vgg19" "dense121"  
# "adv_inception_v3" "ens3_adv_inc_v3" "ens4_adv_inc_v3" "ens_adv_inc_res_v2" 
# "vit_small_patch16_224" "vit_small_patch32_224" "vit_base_patch16_224" "vit_base_patch32_224"
# 'vit_large_patch16_224' "deit_base_patch16_224" "deit3_base_patch16_224" "resnetv2_101x3_bitm"

for source_model in ${source_model_list[@]}
do
    for target_model in ${target_model_list[@]}
    do
        for attack_name in ${attack_name_list[@]}
        do
            device=$(($device+1))
            python /data/huangtao/projects/subsapce-attack/eval/eval.py \
            --attack_name $attack_name \
            --source_model $source_model \
            --device $device \
            --target_model $target_model&
        done
    done
done