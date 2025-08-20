# [Text Embedded Swin-UMamba for DeepLesion Segmentation](https://arxiv.org/abs/2508.06453)

## Key Features

This repository provides the official implementation of: *[Text Embedded Swin-UMamba for DeepLesion Segmentation](https://arxiv.org/abs/2508.06453)*

- Integrate text embedding into the Swin-UMamba decoding path, enable text encoding in nnUNet and its derivatives.
- Curated the image, label, report paired dataset from ULS23 DeepLesion and the original DeepLesion datasets.
- Compared with a few medical imaging segmentation models, Text-Swin-UMamba surpassed the other modes with notable margin in Dice score.
- We open the implementation and dataset for replication.  

## Links

- [Paper](https://github.com/ruida/LLM-Swin-UMamba/blob/main/SPIE_abstract.pdf)
- [Model](https://drive.google.com/file/d/1DtKjVy6ulU2G5c4vLACd6KcWoD6Mx0-V/view?usp=drive_link)
- [Data](https://drive.google.com/drive/folders/1q118BodTfQ3-eVC6ESdPfDdZkDlgvxME?usp=drive_link)
- [Code](https://github.com/ruida/LLM-Swin-UMamba)
- [Results](https://github.com/ruida/LLM-Swin-UMamba/blob/main/results.zip)

## Details

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/architecture.png"></a>
</div>

Segmentation of lesions on CT enables automatic measurement for clinical assessment of chronic diseases (e.g.,
lymphoma). Integrating large language models (LLMs) into the lesion segmentation workflow offers the potential to
combine imaging features with descriptions of lesion characteristics from the radiology reports. In this study, we
investigate the feasibility of integrating text into the Swin-UMamba architecture for the task of lesion segmentation. The
publicly available ULS23 DeepLesion dataset was used along with short-form descriptions of the findings from the reports.
On the test dataset, a high Dice Score of 82 ± 18% and low Hausdorff distance of 6.58 ± 10.64 (pixels) was obtained for
lesion segmentation. The proposed Text-Swin-UMamba model outperformed prior approaches: 37% improvement over
the LLM-driven LanGuideMedSeg model (p < 0.001), and surpassed the purely image-based xLSTM-UNet and nnUNet
models by 1.74% and 0.22%, respectively.



**Main Results**

- Training and validation phase comparison
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/train_dice.png" width="50%" />

- Testing phase compasion
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_dice.png" width="50%" />

- Dice score distribution
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_dice_violin.png" width="50%" />

- Qualitative Comparison
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_result1.png" width="50%" />
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_result2.png" width="50%" />

## Dataset Links

All three datasets can be downloaded from [LLM-Swin-UMamba](https://drive.google.com/drive/folders/1q118BodTfQ3-eVC6ESdPfDdZkDlgvxME?usp=drive_link).

## Get Started

**Main Requirements**  
> torch==2.0.1  
> torchvision==0.15.2  
> causal-conv1d==1.1.1  
> mamba-ssm  
> torchinfo   
> timm  
> numba  


**Installation**
```shell
# create a new conda env
conda create -n swin_umamba python=3.10
conda activate swin_umamba

# install requirements
# Step 1: PyTorch w/ CUDA 11.8
conda install -y -c pytorch -c nvidia pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8

# Step 2: core libs required by Swin-UMamba
pip install causal-conv1d==1.1.1 mamba-ssm==2.2.2 timm==1.0.11 einops==0.8.0 torchinfo==1.8.0

# Step 3: medical imaging + IO stack
pip install monai==1.3.0 nibabel==5.3.2 simpleitk==2.4.0 pydicom==3.0.1 scikit-image==0.24.0 opencv-python==4.10.0.84 imageio==2.36.1 tifffile==2024.9.20 matplotlib==3.9.3 seaborn==0.13.2 pandas==2.2.3 numpy==1.26.4

# Step 4: nnUNetv2 + helpers 
pip install nnunetv2==2.1.1 dynamic-network-architectures==0.3.1 acvl-utils==0.2.2 batchgenerators==0.25.1 connected-components-3d==3.21.0

# Step 5: general utilities
pip install scikit-learn==1.5.2 numba==0.60.0 numexpr==2.10.2 yacs==0.1.8 hiddenlayer==0.3 tqdm==4.67.1 regex==2024.11.6 fsspec==2024.10.0

# Step 6: install the project
cd Swin-UMamba/swin_umamba
pip install -e .
```

**Download Model**

We use the ImageNet pretrained VMamba-Tiny model from [VMamba](https://github.com/MzeroMiko/VMamba). You need to download the model checkpoint and put it into `data/pretrained/vmamba/vmamba_tiny_e292.pth`

```
wget https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth
mv vssmtiny_dp01_ckpt_epoch_292.pth data/pretrained/vmamba/vmamba_tiny_e292.pth
```

**Preprocess**

We use the same data & processing strategy following U-Mamba. Download dataset from [U-Mamba](https://drive.google.com/drive/folders/1q118BodTfQ3-eVC6ESdPfDdZkDlgvxME?usp=drive_link) and put them into the data folder. Then preprocess the dataset with following command:

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 710 --verify_dataset_integrity

(swin_umamba) [me@cn0817 Dataset710_DeepLesion]$ pwd
./Swin-UMamba/data/nnUNet_raw/Dataset710_DeepLesion

(swin_umamba) [me@cn0817 Dataset710_DeepLesion]$ ls
imagesTr labelsTr reports dataset.json 
```


**Training & Testing **

Using the following command to train & test LLM-Swin-UMamba

```shell
conda activate swin_umamba

ml CUDA/11.8
ml gcc/9.2.0

export nnUNet_raw="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_raw"
export nnUNet_preprocessed="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_preprocessed"
export nnUNet_results="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_results"

cd "/data/ruida/segmentation/Swin-UMamba" \
&& nnUNetv2_train 710 2d all -tr nnUNetTrainerSwinUMamba --c 2>&1| tee ${PWD##*/}_train_710.log
```

You can download our model checkpoints [here](https://drive.google.com/file/d/1DtKjVy6ulU2G5c4vLACd6KcWoD6Mx0-V/view?usp=drive_link).

'''shell
nnUNetv2_predict -i ./data/nnUNet_raw/DeepLesion_test/imagesTs -o ./data/nnUNet_raw/DeepLesion_test/prediction -d 710 -tr nnUNetTrainerSwinUMamba -c 2d -f all
'''

## 🙋‍♀️ Feedback and Contact

For further questions, please feel free to contact [Ruida](ruida@nih.gov)


## 🛡️ License

This project is under the Apache License 2.0 license. See [LICENSE](LICENSE) for details.


## 🙏 Acknowledgement
 
Our code is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [UMamba](https://github.com/bowang-lab/U-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet). We thank the authors for making their valuable code & data publicly available.


## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{
    title={Text Embedded Swin-UMamba for DeepLesion Segmentation},
    author={Ruida Cheng, Tejas Mathai, Pritam Mukherjee, Benjamin Hou, Qingqing Zhu, Zhiyong Lu, Matthew McAuliffe, Ronald M. Summers},
    journal={arXiv preprint arXiv:2508.06453},
    year={2025}
}
```
