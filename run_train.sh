#!/bin/bash

### MANUAL STEPS TO DO BEFORE BEGINNING SCRIPT

# authenticate ssh connection
# eval `ssh-agent`
# ssh-add ~/.ssh/id_rsa
ssh tony@10.0.0.26

# ssh tony@ML-A100
# screen -S tony
# rsync -av --delete --exclude='*.git' ~/Documents/git/mha-autoencoder/ tony@ML-A100:~/Documents/git/mha-autoencoder
# WITH CHECKPOINTING
rsync -av --exclude='*.git' ~/Documents/git/mha-autoencoder/ tony@ML-A100:~/Documents/git/mha-autoencoder

### END MANUAL SCRIPTING

# open venv and cd to directory
source ~/Documents/venv/autoencoder/bin/activate
cd ~/Documents/git/mha-autoencoder

# begin training script
python3 train_mha.py \
        --setup_config_path ~/Documents/git/mha-autoencoder/configs/mha_setup.yml \
        --model_config_path ~/Documents/git/mha-autoencoder/configs/model_setup.yml \

# begin inferencing script
python3 test_mha.py \
        --setup_config_path ~/Documents/git/mha-autoencoder/configs/mha_setup.yml \
        --model_config_path ~/Documents/git/mha-autoencoder/configs/model_setup.yml \

# exit gpu server
# detach first
# exit

# copy results back to local machine
scp -r tony@10.0.0.26:~/Documents/training_runs/training_run_7/ /home/tony/Documents/git/mha-autoencoder-results/training_run_7/

# wihtin gpu server
scp -r /home/tony/Documents/git/mha-autoencoder/models/mha/cityscapes/ /home/tony/Documents/training_runs/training_run_13/
scp -r /home/tony/Documents/git/mha-autoencoder/runs/mha/cityscapes/test /home/tony/Documents/training_runs/training_run_13

# copy semseg results tests
scp -r tony@10.0.0.26:/home/tony/Documents/git/mha-autoencoder/models/erfnet/cityscapes/ /home/tony/Documents/git/mha-autoencoder-results/semseg_test

# open tensorboard
tensorboard --host 0.0.0.0 --logdir ./runs
http://ML-A100:6006
http://10.0.0.26:6006

# to delete all epochs except last one
# ls --hide=epoch_25.pth | xargs -d '\n' rm

# size of disk usage
# df -h --total
