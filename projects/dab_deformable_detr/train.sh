srun --gres=gpu:1 --cpus-per-task=16 --qos=preemptive python train_net.py --config-file configs/dab_deformable_detr_12epoch.py --num-gpus 1 --resume
# srun --gres=gpu:4 --cpus-per-task=16 python train_net.py --config-file configs/dab_detr_training.py --num-gpus 4 --resume

export DETECTRON2_DATASETS=/comp_robot/rentianhe/code/IDEADet/datasets
CUDA_VISIBLE_DEVICES=4 python train_net.py --config-file configs/dab_deformable_detr_12epoch.py --num-gpus 1 --resume