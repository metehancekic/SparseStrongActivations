# Sparse Strong Activations

Official repository for the paper entitled ["Neuro-Inspired Deep Neural Networks with Sparse, Strong Activations"](https://arxiv.org/abs/2202.13074 "Metehan Cekic, Can Bakiskan, Upamanyu Madhow"). 

If you have questions you can contact metehancekic [at] ucsb [dot] edu

## Pre-requisites

Install the requirements

> autoattack==0.1
> deepillusion==0.3.2
> foolbox==3.3.1
> Hydra==2.5
> matplotlib==3.3.2
> numpy==1.20.2
> omegaconf==2.1.1
> python-dotenv==0.15.0
> robustbench==1.0
> tensorboard==2.4.1
> torch==1.10.2
> tqdm==4.56.2

Or you can just install by using pip and requirements.txt. Packages like autoattack should be installed through the original repository. Autoattack, robustbench and foolbox are only needed for evaluation.py. If not necessary, you can remove it from the requirements.txt.

```bash
pip install -r requirements.txt
```

## How to install

First, clone the repository to your local machine. Then enter the project folder and create .env file. Then add current directory (project directory) to .env file as following:

```bash
PROJECT_DIR=<project directory>/SparseStrongActivations/
```
put the project directory inside < >.

## Hyper-parameters

All the hyper-parameters can be changed inside the file src/configs/cifar.yaml or through the command calling py file. Default parameters can be found in src/configs/cifar.yaml

```bash
python -m src.train_model --multirun
                train.type=standard
                nn.classifier=HaH_VGG
                nn.conv_layer_type=implicitconv2d
                nn.threshold=-0.2
                nn.divisive.sigma=0.1
                nn.lr=0.001
                train.epochs=100
                train.regularizer.l1_weight.scale=0.001
                train.regularizer.active=['hah']
                train.regularizer.hah.layer=Conv2d
                train.regularizer.hah.alpha=[0.0045,0.0025,0.0013,0.001,0.0008,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
```

## Train a Model

Shell codes inside src/sh are used to train and evaluate the models. To train a HaH model use following shell code:

```bash
bash src/sh/train.sh
```
or simply calling the following command:

```bash
python -m src.train_model
```

## Evaluate a Model

Since we already have a checkpoint, you can run evaluation code on the trained HaH model as follow:

```bash
bash src/sh/eval.sh
```

or simply calling the following command:

```bash
python -m src.eval_model
```

