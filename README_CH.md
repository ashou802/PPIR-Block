# CotIR Block: 使用Chain of Thought 指导高分辨重建进行多尺度融合以补充高频信息
该代码库用于实现 COTB

该代码基于 [SwinIR (PyTorch)](https://github.com/JingyunLiang/SwinIR)构建，并在 Windows 环境（Python 3.10，PyTorch 2.1.2，CUDA 12.1）下使用 3090 GPU 进行测试。


环境库依赖：opencv-python
scikit-image
pillow
torchvision
hdf5storage
ninja
lmdb
requests
timm
einops

## Contents
1. [简介](#简介)
2. [训练](#训练)
3. [测试](#测试)
4. [结果](#结果)
5. [引用](#引用)
6. [致谢](#致谢)

## 简介
超分辨率任务旨在从低分辨率图像恢复出高分辨率图像。当前，大多数超分辨率方法通常包括浅层特征提取、深层特征提取和高分辨率重建三个步骤，而研究的重点往往放在增强深层特征提取能力上。针对高分辨率重建过程中出现的高频信息丢失和全局上下文建模不足的问题，本文提出了一种基于“Chain of Thought（CoT）”思想的多尺度融合模块CotIR Block (COTB)。该模块通过构建多阶段前馈引导路径，逐步融合多尺度特征，有效提升局部细节恢复能力并捕捉长距离语义关联。大量实验结果表明，集成COTB模块的超分辨率网络在多个基准数据集的不同放大倍率下均有不错的性能提升，特别是在复杂纹理场景（Urban100）中，高频细节的保留能力显著增强。通过局部归因图（LAM）和有效感受野（ERF）分析，进一步验证了COTB在扩展模型上下文感知范围、增强远距离区域关联性方面的有效性。此外，COTB模块具有轻量化特点（仅0.57M参数），能够无缝集成至现有超分辨率网络，为高分辨率重建任务提供了高效且可扩展的解决方案。
![COTB](/Figs/COTB.png)
CotIR Block (COTB) 网络结构  
我们将模块的代码整理至`COTB.py`中,同时，为了验证我们模块的性能，将其添加至SwinIR网络中。


## 训练
### 训练数据集准备 

1. 我们采用[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)作为训练集，其中含有800张对应超分倍数（×2、×3、×4、×8）的训练集图片。

2. 修改`main_train_Cotswinir.py`中训练集路径。  

更多的信息你可以参考[SwinIR](https://github.com/JingyunLiang/SwinIR)原文代码或者Github中整理的超分网络集合[KAIR-master](https://github.com/cszn/KAIR)。

### 开始训练

1. 我们提供了COTB应用于Swinir-Light网络中的代码，见文件夹`COTB_with_swinir`，你可以修改其中`main_train_Cotswinir.py`代码中的相关路径直接运行。
2. 如果你不想自己训练，我们将训练好的模型放在了云盘中，你可以直接[下载](https://drive.google.com/file/d/1mWMT0HzM8NbU-dhhGIGN4CNL5VfceA2k/view?usp=sharing)使用训练好的模型进行测试。

    

## 测试
### 快速开始
你可以直接使用我们提供的测试脚本`main_test.py`快速开始测试，只需要更改脚本中测试集路径及超分倍率配置，或者你可以通过下面的控制台指令进行单个测试集测试。

    **以单个测试集Set5为例**

    ```
      python main_test_Cotswinir.py --task lightweight_sr --scale 2 --training_patch_size 48 --model_path model_zoo/SwinirLightwithCOTB_DIV2K_s64w8_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
      python main_test_Cotswinir.py --task lightweight_sr --scale 3 --training_patch_size 48 --model_path model_zoo/SwinirLightwithCOTB_DIV2K_s64w8_x3.pth --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
      python main_test_Cotswinir.py --task lightweight_sr --scale 4 --training_patch_size 48 --model_path model_zoo/SwinirLightwithCOTB_DIV2K_s64w8_x4.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
      python main_test_Cotswinir.py --task lightweight_sr --scale 8 --training_patch_size 48 --model_path model_zoo/SwinirLightwithCOTB_DIV2K_s64w8_x8.pth --folder_lq testsets/Set5/LR_bicubic/X8 --folder_gt testsets/Set5/HR

    ```

## 结果
我们整理了添加COTB模块后在常规任务（X2、X3、X4）在基准测试集上的结果，额外的，我们测试了在更大超分倍率(X8)下的模块的性能，同时进行了局部归因图（LAM）和有效感受野分析（ERF）,实验结果如下：
### 常规任务结果


| Scale | Model                 | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | BSD100 (PSNR/SSIM) | Urban100 (PSNR/SSIM) | Manga109 (PSNR/SSIM) |
|-------|------------------------|------------------|-------------------|--------------------|---------------------|-------------------|
| ×2    | SwinIR-Light           | 38.14 / 0.9611   | 33.86 / 0.9206    | 32.31 / 0.9013     | 32.76 / 0.9340      | 39.11 / 0.9781   |
|       | with COTB              | 38.17 / 0.9612   | 33.95 / 0.9209    | 32.33 / 0.9015     | 32.77 / 0.9342      | 39.07 / 0.9780   |
|       | **promote**            | **+0.03 / +0.0001** | **+0.09 / +0.0003** | **+0.02 / +0.0002** | **+0.01 / +0.0002** | **-0.04 / -0.0001** |
| ×3    | SwinIR-Light           | 34.62 / 0.9289   | 30.54 / 0.8463    | 29.21 / 0.8084     | 28.66 / 0.8624      | 33.99 / 0.9478   |
|       | with COTB              | 34.66 / 0.9292   | 30.55 / 0.8466    | 29.23 / 0.8089     | 28.72 / 0.8636      | 34.07 / 0.9483   |
|       | **promote**            | **+0.04 / +0.0003** | **+0.01 / +0.0003** | **+0.02 / +0.0005** | **+0.06 / +0.0012** | **+0.08 / +0.0005** |
| ×4    | SwinIR-Light           | 32.44 / 0.8976   | 28.77 / 0.7858    | 27.70 / 0.7408     | 26.47 / 0.7980      | 30.92 / 0.9150   |
|       | with COTB              | 32.46 / 0.8981   | 28.78 / 0.7860    | 27.71 / 0.7412     | 26.51 / 0.7990      | 30.97 / 0.9156   |
|       | **promote**            | **+0.02 / +0.0005** | **+0.01 / +0.0004** | **+0.01 / +0.0004** | **+0.04 / +0.0010** | **+0.05 / +0.0006** |

**实际纹理效果图**  

![Practical_results](/Figs/Practical results.png)
### 8倍超分倍率结果

| Test Sets  | Model                      | SwinIR-Light | SwinIR-Light with COTB |
|------------|----------------------------|--------------|------------------------|
| Set5       | (PSNR/SSIM)                 | 27.11 / 0.7790 | 27.15 / 0.7790 |
| Set14      | (PSNR/SSIM)                 | 25.05 / 0.6448 | 25.11 / 0.6459 |
| BSD100     | (PSNR/SSIM)                 | 24.88 / 0.6006 | 24.88 / 0.6012 |
| Urban100   | (PSNR/SSIM)                 | 22.65 / 0.6260 | 22.65 / 0.6263 |
| Manga109   | (PSNR/SSIM)                 | 24.79 / 0.7876 | 24.83 / 0.7874 |

### 不同特征提取方法结果

| Test Sets | Model                                      | SwinIR-Light | SwinIR-Light with Global feature | SwinIR-Light with Channel feature |
|-----------|-------------------------------------------|--------------|----------------------------------|-----------------------------------|
| Set5      | (PSNR/SSIM)                               | 32.44 / 0.8976 | 32.46 / 0.8981 | 32.49 / 0.8983 |
| Set14     | (PSNR/SSIM)                               | 28.77 / 0.7858 | 28.78 / 0.7860 | 28.78 / 0.7861 |
| BSD100    | (PSNR/SSIM)                               | 27.70 / 0.7408 | 27.71 / 0.7412 | 27.71 / 0.7412 |
| Urban100  | (PSNR/SSIM)                               | 26.47 / 0.7980 | 26.51 / 0.7990 | 26.51 / 0.7992 |
| Manga109  | (PSNR/SSIM)                               | 30.92 / 0.9150 | 30.97 / 0.9156 | 30.99 / 0.9158 |

### LAM分析

![LAM](/Figs/LAM.png)

### ERF分析

![ERF](/Figs/ERF.png)

## 引用
如果您在研究或工作中发现此代码有帮助，请引用以下论文。

```
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```
## 致谢
这项目代码基于 [SwinIR](https://github.com/JingyunLiang/SwinIR) 和 [KAIR-master](https://github.com/cszn/KAIR) 构建，感谢作者分享他们的代码。
