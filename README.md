# PPIR Block: generate prior prompt information to dynamically adjust the high-resolution reconstruction process
This repository is for PPIRB introduced in the following paper


The code is built on [SwinIR (PyTorch)](https://github.com/JingyunLiang/SwinIR) and tested on 
Windows environment (Python3.10, PyTorch_2.1.2, cu121) with 3090 GPUs.   

requirements:opencv-python
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
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#Citation)
5. [Acknowledgements](#acknowledgements)

## Introduction
The goal of super‑resolution is to recover a high‑resolution (HR) image from its low‑resolution (LR) counterpart. Most existing SR approaches decompose the task into two stages: feature extraction and HR reconstruction, with a focus on enhancing feature extraction capabilities. To address the issues of high‑frequency information loss and insufficient global context modeling in high-resolution reconstruction, this paper proposes a multi‑scale fusion inference module called Priori Prompt Image Restoration Block (PPIRB), based on the “result prompt process”. In PPIRB, we adopt a step-by-step approach of adding guidance information to achieve hierarchical inference in the reconstruction process. For the generation of guidance information, a prior method is innovatively introduced to generate an approximate representation of the HR image in advance before reconstructing it, to prompt the original reconstruction process and achieve a dynamically adjusted enhancement strategy. By enhancing long-range semantic dependencies, this strategy significantly improves the recovery of high-frequency details and local structures. Extensive experimental results have shown that the super-resolution network integrated with PPIRB module has good performance improvement at different magnifications of multiple benchmark datasets, especially in complex texture scenes (such as Urban100), where the ability to preserve high-frequency details is significantly enhanced. The effectiveness of PPIRB in expanding the context aware range of the model and enhancing long-range spatial dependencies was further validated through Local Attribution Maps (LAM) and Effective Receptive Field (ERF) analysis. In addition, PPIRB is a lightweight module (with only 0.57M parameters), and it does not change the input and output of the original reconstruction process, nor does it damage the feature extraction part of the original model. It can be seamlessly integrated into or used as a direct replacement for reconstruction modules in existing super-resolution networks.

![PPIRB](/Figs/PPIRB.png)
The architecture of  PPIR Block (PPIRB)  

We have organized the code for the module into `PPIRB.py`. Additionally, to validate the performance of our module, we have integrated it into the SwinIR network.

## Train
### Prepare training data 

1. We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as the training dataset, which contains 800 images corresponding to super-resolution scales (×2, ×3, ×4, ×8).
2. Modify the training dataset path in main_train_PPIR_swinir.py.

For more information, you can refer to the original [SwinIR](https://github.com/JingyunLiang/SwinIR)code or the super-resolution network collection [KAIR-master](https://github.com/cszn/KAIR) on GitHub.

### Begin to train

1. We provide the code for applying PPIRB to the SwinIR-Light network, which can be found in the PPIRB_with_swinir folder. You can modify the relevant paths in main_train_PPIR_swinir.py and run it directly.
2. If you don't want to train the model yourself, we have uploaded the pre-trained model to the cloud. You can directly [download](https://drive.google.com/file/d/1mWMT0HzM8NbU-dhhGIGN4CNL5VfceA2k/view?usp=sharing) and use it for testing.

    

## Test
### Quick start
You can directly use our provided test script, `main_test.py`, to quickly start testing. Simply modify the test dataset path and super-resolution scale configuration in the script. Alternatively, you can run a single test on a dataset using the following console command.  

    **As an example, for a single test dataset Set5, you can run the following command in the console**

    ```
      python main_test_PPIR_swinir.py --task lightweight_sr --scale 2 --training_patch_size 48 --model_path model_zoo/SwinirLightwithPPIRB_DIV2K_s64w8_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
      python main_test_PPIR_swinir.py --task lightweight_sr --scale 3 --training_patch_size 48 --model_path model_zoo/SwinirLightwithPPIRB_DIV2K_s64w8_x3.pth --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
      python main_test_PPIR_swinir.py --task lightweight_sr --scale 4 --training_patch_size 48 --model_path model_zoo/SwinirLightwithPPIRB_DIV2K_s64w8_x4.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
      python main_test_PPIR_swinir.py --task lightweight_sr --scale 8 --training_patch_size 48 --model_path model_zoo/SwinirLightwithPPIRB_DIV2K_s64w8_x8.pth --folder_lq testsets/Set5/LR_bicubic/X8 --folder_gt testsets/Set5/HR

    ```

## Results
We have organized the results of adding the PPIRB module on the benchmark dataset for standard tasks (X2, X3, X4). Additionally, we tested the performance of the module at a larger upscaling factor (X8) and conducted local attribution maps (LAM) and effective receptive field (ERF) analysis. The experimental results are as follows:
### Classical task results


| Scale | Model                 | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | BSD100 (PSNR/SSIM) | Urban100 (PSNR/SSIM) | Manga109 (PSNR/SSIM) |
|-------|------------------------|------------------|-------------------|--------------------|---------------------|-------------------|
| ×2    | SwinIR-Light           | 38.14 / 0.9611   | 33.86 / 0.9206    | 32.31 / 0.9013     | 32.76 / 0.9340      | 39.11 / 0.9781   |
|       | with PPIRB              | 38.16 / 0.9612   | 33.90 / 0.9207    | 32.32 / 0.9014     | 32.81 / 0.9344      | 39.18 / 0.9782   |
|       | **promote**            | **+0.02 / +0.0001** | **+0.04 / +0.0001** | **+0.01 / +0.0001** | **+0.05 / +0.0004** | **+0.07 / +0.0001** |
| ×3    | SwinIR-Light           | 34.62 / 0.9289   | 30.54 / 0.8463    | 29.21 / 0.8084     | 28.66 / 0.8624      | 33.99 / 0.9478   |
|       | with PPIRB              | 34.66 / 0.9292   | 30.55 / 0.8466    | 29.23 / 0.8089     | 28.72 / 0.8636      | 34.07 / 0.9483   |
|       | **promote**            | **+0.04 / +0.0003** | **+0.01 / +0.0003** | **+0.02 / +0.0005** | **+0.06 / +0.0012** | **+0.08 / +0.0005** |
| ×4    | SwinIR-Light           | 32.44 / 0.8976   | 28.77 / 0.7858    | 27.70 / 0.7408     | 26.47 / 0.7980      | 30.92 / 0.9150   |
|       | with PPIRB              | 32.46 / 0.8981   | 28.78 / 0.7860    | 27.71 / 0.7412     | 26.51 / 0.7990      | 30.97 / 0.9156   |
|       | **promote**            | **+0.02 / +0.0005** | **+0.01 / +0.0004** | **+0.01 / +0.0004** | **+0.04 / +0.0010** | **+0.05 / +0.0006** |

**Renderings of actual textures**  

![Practical_results](/Figs/Practical results.png)
### 8x super-resolution magnification results

| Test Sets  | Model                      | SwinIR-Light | SwinIR-Light with PPIRB |
|------------|----------------------------|--------------|------------------------|
| Set5       | (PSNR/SSIM)                 | 27.11 / 0.7790 | 27.15 / 0.7790 |
| Set14      | (PSNR/SSIM)                 | 25.05 / 0.6448 | 25.11 / 0.6459 |
| BSD100     | (PSNR/SSIM)                 | 24.88 / 0.6006 | 24.88 / 0.6012 |
| Urban100   | (PSNR/SSIM)                 | 22.65 / 0.6260 | 22.65 / 0.6263 |
| Manga109   | (PSNR/SSIM)                 | 24.79 / 0.7876 | 24.83 / 0.7874 |

### Results of different feature extraction methods

| Test Sets | Model                                      | SwinIR-Light | SwinIR-Light with Global feature | SwinIR-Light with Channel feature |
|-----------|-------------------------------------------|--------------|----------------------------------|-----------------------------------|
| Set5      | (PSNR/SSIM)                               | 32.44 / 0.8976 | 32.46 / 0.8981 | 32.49 / 0.8983 |
| Set14     | (PSNR/SSIM)                               | 28.77 / 0.7858 | 28.78 / 0.7860 | 28.78 / 0.7861 |
| BSD100    | (PSNR/SSIM)                               | 27.70 / 0.7408 | 27.71 / 0.7412 | 27.71 / 0.7412 |
| Urban100  | (PSNR/SSIM)                               | 26.47 / 0.7980 | 26.51 / 0.7990 | 26.51 / 0.7992 |
| Manga109  | (PSNR/SSIM)                               | 30.92 / 0.9150 | 30.97 / 0.9156 | 30.99 / 0.9158 |

### LAM Analyse

![LAM](/Figs/LAM.png)

### ERF Analyse

![ERF](/Figs/ERF.png)



## Acknowledgements
This code is built on  [SwinIR](https://github.com/JingyunLiang/SwinIR) and [KAIR-master](https://github.com/cszn/KAIR). We thank the authors for sharing their codes.
