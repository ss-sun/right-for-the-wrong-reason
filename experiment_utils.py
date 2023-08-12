from __future__ import print_function
import os
import shutil
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import wandb
import datetime
import json


def update_key_value_pairs(filename, model_name, measure_name, value):
    with open(filename, "r") as file:
        data = json.load(file)
    dict = data[model_name]
    dict[measure_name] = value
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def init_seed(manual_seed):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    cudnn.benchmark = True


def create_expname(exp_configs):
    current_time = datetime.datetime.now()
    if exp_configs.exp_name == "resnet_cls":
        exp_name = exp_configs.exp_name + str(current_time)[:-7]
        exp_name += f"-{exp_configs.dataset}"
        exp_name += f"-bs={exp_configs.batch_size}"
        exp_name += f"-lr={exp_configs.lr}"
        exp_name += f"-weight_decay={exp_configs.weight_decay}"

    if exp_configs.exp_name == "attrinet":
        exp_name = exp_configs.exp_name + str(current_time)[:-7]
        exp_name += f"--{exp_configs.dataset}"
        exp_name += f"--bs={exp_configs.batch_size}"
        exp_name += f"--lg_ds={exp_configs.lgs_downsample_ratio}"
        exp_name += f"--l_cri={exp_configs.lambda_critic}"
        exp_name += f"--l1={exp_configs.lambda_1}"
        exp_name += f"--l2={exp_configs.lambda_2}"
        exp_name += f"--l3={exp_configs.lambda_3}"
        exp_name += f"--l_ctr={exp_configs.lambda_centerloss}"

    exp_configs.exp_name = exp_name



def init_experiment(exp_configs):
    create_expname(exp_configs)
    print('exp_configs.exp_name', exp_configs.exp_name)
    os.makedirs(exp_configs.save_path, exist_ok=True)
    exp_configs.exp_dir = os.path.join(exp_configs.save_path, exp_configs.exp_name)
    exp_configs.ckpt_dir = exp_configs.exp_dir + '/ckpt'
    exp_configs.output_dir = exp_configs.exp_dir + '/output'

    for path in [exp_configs.exp_dir, exp_configs.ckpt_dir, exp_configs.output_dir]:
        try:
            shutil.rmtree(path)
        except:
            pass
        os.makedirs(path)



def init_wandb(exp_configs):
    wandb.login(key='your key')
    wandb.init(dir=exp_configs.save_path,
               project="right_for_wrong_reasons",
               name = exp_configs.exp_name,
               notes='train on' + exp_configs.dataset,
               )

    config = wandb.config
    if "resnet" in exp_configs.exp_name:
        config.batch_size = exp_configs.batch_size
        config.lr = exp_configs.lr
    if "attrinet" in exp_configs.exp_name:
        config.logreg_mode = exp_configs.lgs_downsample_ratio
        config.lambda_critic = exp_configs.lambda_critic
        config.lambda_1 = exp_configs.lambda_1
        config.lambda_2 = exp_configs.lambda_2
        config.lambda_3 = exp_configs.lambda_3
        config.lambda_centerloss = exp_configs.lambda_centerloss

    wandb.run.save()


