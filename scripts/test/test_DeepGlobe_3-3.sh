DATA_ROOT=./data/DeepGlobe2018
DATASET=deepglobe
TASK=3-3
EPOCH=30
BATCH=12
LOSS=focal_loss
LR=0.01
THRESH=0.7
MEMORY=0
step=0

python eval.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze   \
    --w_transfer --amp --mem_size ${MEMORY} --curr_step ${step} --crop_size 1024 --infer_branch ce
