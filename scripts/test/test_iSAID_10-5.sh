DATA_ROOT=../autodl-tmp/iSAID
DATASET=iSAID
TASK=10-5
EPOCH=30
BATCH=30
LOSS=focal_loss
LR=0.01
THRESH=0.4
MEMORY=0
step=0

python eval.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --w_transfer --amp --mem_size ${MEMORY} --curr_step ${step} --crop_size 1024 --infer_branch ce\
    --val_batch_size 4 --test_on_val
