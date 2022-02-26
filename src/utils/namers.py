import torch
import os
import numpy as np

from omegaconf import DictConfig


def classifier_params_string(model_name: str, cfg: DictConfig):
    classifier_params_string = model_name

    classifier_params_string += f"_lr_{cfg.nn.lr:.4f}"

    if cfg.train.type == "noisy":
        classifier_params_string += f"_noisytr_{'_'.join([str(x) for x in cfg.train.noise.values()])}"

    if cfg.train.type == "adversarial":
        classifier_params_string += f"_advtr_{'_'.join([str(x) for x in cfg.train.adversarial.values()])}"

    if "l1_weight" in cfg.train.regularizer.active:
        classifier_params_string += f"_l1_weight_{'_'.join([str(x) for x in cfg.train.regularizer.l1_weight.values()])}"

    if "hah" in cfg.train.regularizer.active:
        classifier_params_string += f"_hah"
        classifier_params_string += f"_{'_'.join([str(x) for x in cfg.train.regularizer.hah.values()])}"

    classifier_params_string += f"_ep_{cfg.train.epochs}"
    classifier_params_string += f"_seed_{cfg.seed}"

    return classifier_params_string


def classifier_ckpt_namer(model_name: str, cfg: DictConfig):

    file_path = cfg.directory + f"checkpoints/{cfg.dataset.name}/"
    os.makedirs(file_path, exist_ok=True)

    file_path += classifier_params_string(model_name, cfg)

    file_path += ".pt"

    return file_path


def classifier_log_namer(model_name: str, cfg: DictConfig):

    file_path = cfg.directory + f"logs/{cfg.dataset.name}/"

    os.makedirs(file_path, exist_ok=True)

    file_path += classifier_params_string(model_name, cfg)

    file_path += ".log"

    return file_path
