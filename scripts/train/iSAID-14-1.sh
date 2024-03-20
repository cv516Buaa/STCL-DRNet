DATA_ROOT=../autodl-tmp/iSAID
DATASET=iSAID
TASK=14-1
EPOCH=30
BATCH=30
LOSS=focal_loss
LR=0.01
THRESH=0.4
MEMORY=0
step=0

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --lr ${LR} --crop_val\
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --print_interval 300 --val_interval 600\
    --w_transfer --amp --mem_size ${MEMORY} --curr_step ${step} --crop_size 512\
    --train_branch ce --infer_branch ce --kd 2 --dkd 2 --test_on_val