# models=(tf_resnet_v2_50 tf_inception_v3 tf_inc_res_v2)
source_model=tf_inception_v3

models=(vgg19)
for model in "${models[@]}"
do
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 --preemptible yes -- python -m plots.pca2 \
    --source_model ${source_model} \
    --target_model ${model} \
    --contourf \
    --proj cos \
    --N 51 &
done
