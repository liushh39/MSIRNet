<h1 align="center">MSIRNet: Learning multi-granularity semantic interactive representation for joint low-light image enhancement and super-resolution</h1>

<div align='center'>
    <a target='_blank'><strong>Jing Ye</strong></a><sup> 1</sup>&emsp;
    <a href='https://github.com/liushh39' target='_blank'><strong>Shenghao Liu</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Changzhen Qiu</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Zhiyong Zhang</strong></a><sup> 1†</sup>&emsp;
    
</div>

<div align='center'>
    <sup>1 </sup>Sun Yat-Sen University&emsp; <small><sup>†</sup> Corresponding author</small>;
</div>

<h1 align="center"><img src="https://github.com/liushh39/MSIRNet/blob/main/framework.png" width="1000"></h1>

## Introduction 📖
This repo, named **MSIRNet**, contains the official PyTorch implementation of our paper [Learning multi-granularity semantic interactive representation for joint low-light image enhancement and super-resolution](https://www.sciencedirect.com/science/article/pii/S1566253524002458).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.


## Dependencies and Installation

```
# create new anaconda env
conda create -n msir python=3.7.11
conda activate msir 

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```


## Dataset

- Download the [dataset](https://pan.baidu.com/s/1d7bO9lZrbpbxoX-Zl7ub2w?pwd=lyc9).
- Specify their path in the corresponding option file or extract it to the project root directory.

## Quick Inference
- Download our [model](https://github.com/liushh39/MSIRNet/releases/download/v1.0.0/net_g_111250.pth)
- Put the pretrained models in `experiments/`

```
python inference_MSIRNet.py
```

## Train the model

### Model preparation

Before training, you need to
- Download the pretrained HRP model: [generator](https://github.com/liushh39/MSIRNet/releases/download/v1.0.0/FeMaSR_HRP_model_g.pth), [discriminator](https://github.com/liushh39/MSIRNet/releases/download/v1.0.0/FeMaSR_HRP_model_d.pth) 
- Put the pretrained models in `experiments/pretrained_models`
- Specify their path in the corresponding option file.

### Train SR model

```
python basicsr/train.py -opt options/train_MSIR_LQ_stage_LOLX4.yml
```

## Contact Informaiton
If you have any questions, please feel free to contact me at liushh39@mail2.sysu.edu.cn.

## Citation
```
@article{YE2024102467,
title = {Learning multi-granularity semantic interactive representation for joint low-light image enhancement and super-resolution},
journal = {Information Fusion},
pages = {102467},
year = {2024},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2024.102467},
url = {https://www.sciencedirect.com/science/article/pii/S1566253524002458},
author = {Jing Ye and Shenghao Liu and Changzhen Qiu and Zhiyong Zhang},
}
```

## Acknowledgement

The code is based on [FeMaSR](https://github.com/chaofengc/FeMaSR) and [BasicSR](https://github.com/xinntao/BasicSR).
