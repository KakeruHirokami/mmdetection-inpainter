# Overview
This repository provides functionality to inpaint specific objects in video using a mmdetection.

# Requirements
This repository requires the following tools:
- windows
- CUDA_11.8
- python3.10.X
- ffmpeg
- Miniconda

It may work with other versions, but it is not guaranteed.

# Installation
## Install ffmpeg
This repository uses ffmpeg command.  
Please install from the official FFmpeg website.  

## Install cuda11.8
### Chack NVIDIA Driver

Execute following command and confirm CUDA version is 11.8 or higher.
```
$ nvidia-smi
```

### Check CUDA Toolkit
Execute following command and confirm CUDA version is 11.8 or higher.  
Verify that CUDA_11.8 is built.
```
$ nvcc -V
```

## Install Miniconda
Download Miniconda in official site
https://docs.anaconda.com/miniconda/

## Create MMDetection env
Refer MMDetection official site.
https://mmdetection.readthedocs.io/en/latest/get_started.html

> [!NOTE]
> This document is for OS X.
> Pytorch install command is a bit different from the command mentioned in this document.
> Quote from Pytorch official site.
> https://pytorch.org/get-started/previous-versions/

> [!NOTE]
> As of July 23, 2024, MMDetection appeared to does not supported mmcv >= 2.2.0.
> So, mmcv version you should install is 2.1.0, and mmcv 2.1.0 is only supported 1.8.X <= torch <= 2.1.X.
> To install MMDetection, it is better to install torch anb mmcv using the following commands.
> `conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda={your cuda version} -c pytorch -c nvidia`
> `pip install -U openmim`
> `mim install "mmcv==2.1.0"`

# Quick start
## Download config and checkpoint files
Following the document, download R-50-FPN pytorch 2x model.
https://github.com/open-mmlab/mmdetection/tree/main/configs/mask_rcnn

## Execute
Execute following command.
```
$ python main.py c
```