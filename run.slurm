#!/bin/sh
#SBATCH -J train
#SBATCH -o output/log.%j.out
#SBATCH -e output/log.%j.err
#SBATCH --partition=gpu3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --gpus=1
#SBATCH --mem=12G

# 激活lilei环境
. /home/liqinglei/miniconda3/etc/profile.d/conda.sh
conda activate Lilei

# 显示环境信息
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Job started at: $(date)"
echo "Running on: $(hostname)"

# python DRC_TCO.py --batch_size 32 --nw 10 --tco 1 --threshold_acc 100 --epochs 30 --dataset 'CUB200' --confidence 0.99 --lr 5e-3 --weight_decay 1e-4 --lambda_aug 0.3 --rewind_threshold 5 --epoch_start 4

# python baseline_pseudo_label.py --batch_size 64 --nw 10 --epochs 30 --dataset 'CUB200' --confidence 0.9 --lr 5e-3 --weight_decay 1e-4

# python baseline_fixmatch.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' --confidence_threshold 0.7 --lambda_u 1.0

# python baseline_tent.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' --tent_lr 1e-3 --tent_steps 1

# python baseline_flexmatch.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 

# python baseline_freematch.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 

# python baseline_mean_teacher.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 

# python baseline_cotta.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 

python baseline_sar.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 

# python baseline_simclr.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 

# python baseline_moco.py --batch_size 32 --nw 10 --epochs 30 --dataset 'CUB200' 