# [LLM Embedded Swin-UMamba for DeepLesion Segmentation](https://arxiv.org/abs/2402.03302)

## Key Features

This repository provides the official implementation of: *[LLM Embedded Swin-UMamba for DeepLesion Segmentation](https://arxiv.org/abs/2402.03302)*

- Integrate LLM text embedding into the Swin-UMamba decoding path, enable text encoding in nnUNet.
- Curated the image, label, report paired dataset from ULS23 DeepLesion and the original DeepLesion datasets.
- Compared with a few medical imaging segmentation models, LLM-Swin-UMamba surpassed the other modes with notable margin in Dice score.
- We open the implementation and dataset for replication.  

## Links

- [Paper](https://arxiv.org/abs/2402.03302)
- [Model](https://drive.google.com/file/d/1DtKjVy6ulU2G5c4vLACd6KcWoD6Mx0-V/view?usp=drive_link)
- [Data](https://drive.google.com/drive/folders/1q118BodTfQ3-eVC6ESdPfDdZkDlgvxME?usp=drive_link)
- [Code](https://github.com/ruida/LLM-Swin-UMamba)
- [Results](https://github.com/ruida/LLM-Swin-UMamba/blob/main/results.zip)

## Details

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/architecture.png"></a>
</div>

Recent rapid developments in Large Language Models (LLMs) bring the potential to integrating LLM into medical image segmentation. Our previous studies have applied universal lesion detection to the DeepLesion dataset, achieving state-of-the-art accuracy in lesion detection and short form report prediction tasks with CNN architectures. In this study, we investigate the feasibility of integrating the LLM model into the Swin-UMamba architecture to segment lesions using the DeepLesion dataset. We combine the DeepLesion short form report finding with the ULS23 DeepLesion dataset to conduct the lesion segmentation task. We achieved relatively high segmentation performance with a Dice score of 81.76% in the testing phase.  We also compare segmentation performance with a few LLM-driven medical image segmentation models. The proposed LLM-Swin-UMamba model outperforms the other models in mean Dice Score.  In this work, we demonstrate the feasibility of integrating LLM into the DeepLesion segmentation task. 



**Main Results**

- Training and validation phase comparison
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/train_dice.png" width="50%" />

- Testing phase compasion
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_dice.png" width="50%" />

- Dice score distribution
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_dice_violin.png" width="50%" />

- Qualitative Comparison
<img src="https://github.com/ruida/LLM-Swin-UMamba/blob/main/assets/test_result.png" width="50%" />

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
pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install torchinfo timm numba

# install swin_umamba
git clone https://github.com/JiarunLiu/Swin-UMamba
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

We use the same data & processing strategy following U-Mamba. Download dataset from [U-Mamba](https://github.com/bowang-lab/U-Mamba) and put them into the data folder. Then preprocess the dataset with following command:

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```


**Training & Testing**

Using the following command to train & test Swin-UMamba

```shell
# AbdomenMR dataset
bash scripts/train_AbdomenMR.sh MODEL_NAME
# Endoscopy dataset
bash scripts/train_Endoscopy.sh MODEL_NAME
# Microscopy dataset 
bash scripts/train_Microscopy.sh MODEL_NAME
```

Here  `MODEL_NAME` can be:

- `nnUNetTrainerSwinUMamba`: Swin-UMamba model with ImageNet pretraining
- `nnUNetTrainerSwinUMambaD`: Swin-UMamba$\dagger$  model with ImageNet pretraining
- `nnUNetTrainerSwinUMambaScratch`: Swin-UMamba model without ImageNet pretraining
- `nnUNetTrainerSwinUMambaDScratch`: Swin-UMamba$\dagger$  model without ImageNet pretraining

You can download our model checkpoints [here](https://drive.google.com/drive/folders/1zOt0ZfQPjoPdY37NfLKevYs4x5eClThN?usp=sharing).


## 🙋‍♀️ Feedback and Contact

For further questions, please feel free to contact [Jiarun Liu](jr.liu@siat.ac.cn)


## 🛡️ License

This project is under the Apache License 2.0 license. See [LICENSE](LICENSE) for details.


## 🙏 Acknowledgement
 
Our code is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [UMamba](https://github.com/bowang-lab/U-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet). We thank the authors for making their valuable code & data publicly available.


## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{Swin-UMamba,
    title={Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining},
    author={Jiarun Liu and Hao Yang and Hong-Yu Zhou and Yan Xi and Lequan Yu and Yizhou Yu and Yong Liang and Guangming Shi and Shaoting Zhang and Hairong Zheng and Shanshan Wang},
    journal={arXiv preprint arXiv:2402.03302},
    year={2024}
}
```
