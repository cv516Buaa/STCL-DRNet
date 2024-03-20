DATA_ROOT=./data/DeepGlobe2018
DATASET=deepglobe
TASK=1-1
EPOCH=50
BATCH=8
LOSS=focal_loss
LR=0.01
THRESH=0.4
MEMORY=0
step=0

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --lr ${LR} --crop_val\
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --w_transfer --amp --mem_size ${MEMORY} --curr_step ${step} --crop_size 1024 --train_branch ce --infer_branch ce --kd 2
