#!/bin/bash
#SBATCH --job-name=edward-pigan

#SBATCH --partition=rtx6000

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# Create Checkpoint Directory
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYEDPURGE
# Local symbolic link
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint

# prepare your environment here
# module load pytorch1.7.1-cuda11.0-python3.6
module purge
source /h/edwardl/pigan/pigan_env/bin/activate

# put your command here
# python train.py
CUDA_VISIBLE_DEVICES=0 python3 train.py --curriculum CARLA --output_dir $PWD/checkpoint --load_dir $PWD/checkpoint --model_save_interval 50 --n_epochs 200

# copy over checkpoint files
cp -r /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYEDPURGE/ /h/edwardl/pigan/${SLURM_JOB_ID}/output/