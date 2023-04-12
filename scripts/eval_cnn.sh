device=1
beta_list=(2)
# 2 2 2 2)
lr_list=(0.000005)
iter_num_list=(10)
N_list=(10)
N_vt_list=(20)
beta_vt_list=(1.5)
attack_name_list=('vnim')
#'mim_sub_line')
source_model_list=("resnet_v2_152")
#  'resnet_v2_152' 'resnet_v2_101' 'inception_v3'  "resnet_v2_50" "inc_res_v2" "adv_inception_v3" "ens3_adv_inc_v3" "ens4_adv_inc_v3" "ens_adv_inc_res_v2"
target_model_list=("inc_res_v2" "adv_inception_v3" "ens3_adv_inc_v3" "ens4_adv_inc_v3" "ens_adv_inc_res_v2")
for source_model in ${source_model_list[@]}
do
    for target_model in ${target_model_list[@]}
    do
        for attack_name in ${attack_name_list[@]}
        do
            for beta in ${beta_list[@]}
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
                                    device=$(($device))
                                    python /data/huangtao/projects/subsapce-attack/eval/eval_cnn.py \
                                    --attack_name $attack_name \
                                    --beta $beta \
                                    --source_model $source_model \
                                    --target_model $target_model \
                                    --iter_num $iter_num \
                                    --device $device \
                                    --N $N \
                                    --iter_num $iter_num \
                                    --lr $lr \
                                    --N_vt $N_vt \
                                    --beta_vt $beta_vt&
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done