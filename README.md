# [Text Embedded Swin-UMamba for DeepLesion Segmentation](https://arxiv.org/abs/2508.06453)

## Key Features

This repository provides the official implementation of: *[Text Embedded Swin-UMamba for DeepLesion Segmentation](https://github.com/ruida/LLM-Swin-UMamba/blob/main/SPIE_paper.pdf)*

- Integrate text embedding into the Swin-UMamba decoding path, enable text encoding in nnUNet and its derivatives.
- Curated the image, label, report paired dataset from ULS23 DeepLesion and the original DeepLesion datasets.
- Compared with a few medical imaging segmentation models, Text-Swin-UMamba surpassed the other models with notable margin in Dice score.
- We open the implementation and dataset for replication.  

## Links

- [Paper](https://github.com/ruida/LLM-Swin-UMamba/blob/main/SPIE_paper.pdf)
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
On the test dataset, a high Dice Score of 82.64 ± 17.36 % and low Hausdorff distance of 6.34 ± 10.48 (pixels) was obtained for
lesion segmentation. The proposed Text-Swin-UMamba model outperformed prior approaches: 37.79 % improvement over
the LLM-driven LanGuideMedSeg model (p < 0.001), and surpassed the purely image-based xLSTM-UNet and nnUNet
models by 2.58 % and 1.01 %, respectively.



**Main Results**

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

**Installation**
```shell
# create a new conda env
conda create -n llm-swin_umamba python=3.10
conda activate llm-swin_umamba

# install requirements
# Step 1: PyTorch w/ CUDA 12.1

ml CUDA/12.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Step 2: core libs required by Swin-UMamba
pip install causal-conv1d==1.1.1

pip install  markdown>=2.6.8

pip install protobuf!=4.24.0,>=3.19.6

pip install mamba-ssm==2.2.2

pip install numpy==1.26.4

pip install opencv-python==4.10.0.84

pip install transformers==4.41.2

pip install torchinfo timm numba

cd swin_umamba
pip install -e .
cd ..
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

ml CUDA/12.1
ml gcc/9.2.0

export nnUNet_raw="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_raw"
export nnUNet_preprocessed="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_preprocessed"
export nnUNet_results="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_results"

cd "/data/ruida/segmentation/Swin-UMamba" \
&& nnUNetv2_train 710 2d all -tr nnUNetTrainerSwinUMamba --c 2>&1| tee ${PWD##*/}_train_710.log
```

You can download our model checkpoints [here](https://drive.google.com/file/d/1DtKjVy6ulU2G5c4vLACd6KcWoD6Mx0-V/view?usp=drive_link).

```shell
nnUNetv2_predict -i ./data/nnUNet_raw/DeepLesion_test/imagesTs -o ./data/nnUNet_raw/DeepLesion_test/prediction -d 710 -tr nnUNetTrainerSwinUMamba -c 2d -f all
```

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
