# LDI-Seg


Accurate segmentation of medical images remains challenging due to the significant variations in lesion size, ambiguous boundaries, and severe noise interference inherent in clinical imaging. To address these issues, this study revisits feature fusion strategies from a frequency-domain perspective and proposes a novel segmentation model named LDI-Seg.
A Multi-Level Frequency Complementation (MFC) module is proposed to fuse high-level and low-level encoder features in a frequency-differentiated manner. Specifically, it enhances shallow structural representations with deep semantic guidance while adaptively suppressing high-frequency noise through mid-level feature guided gating. A Decoder Frequency-enhanced Edge-aware (DFE) module is designed to optimize encoder-decoder feature fusion via wavelet-enhanced channel attention and edge prior guidance, explicitly leveraging encoder high-frequency details to refine decoder features. LDI-Seg integrates these two modules with a pre-trained InceptionNeXt encoder.
To verify the effectiveness of the proposed modules, extensive experiments are conducted on four public datasets (ISIC2018, BUSI, Kvasir-SEG, and CVC-ClinicDB) as well as a private Lung-CT dataset. The results demonstrate that LDI-Seg consistently outperforms state-of-the-art methods across multiple evaluation metrics. The source code is available at: https://github.com/shen123shen/LDI-Seg. The source code is available at: https://github.com/shen123shen/LDI-Seg.
# Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:\
ISIC-2018 (dermoscopy, 2,594 images fortraining, 100 images for validation, and 1,000 images for testing)\
Kvasir-SEG (gastrointestinal polyp, 600 images for training, 200images for validation, and 200 images for testing)\
BUSI (breast ultrasound, 399 images for training.113 images for validation, and 118 images for testing)\
CVC-ClinicDB (colorectal cancer, 367 images for training, 123images for validation, and 122 images for testing)\
Lung-CT (Chest CT, 210 images for training, 45images for validation, and 45 images for testing)\
The dataset path may look like:
```
/The Dataset Path/
├── ISIC-2018/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```
 # Usage
 Installation
 ```
 git clone git@github.com:shen123shen/LDI-Seg.git
 conda create -n shen python=3.8
 conda activate shen
 conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Training
 ```
python train_cuda.py
 ```
Evaluation
 ```
python Test.py
 ```
# Citation

 ```
@ARTICLE{40030292,
  author  = {Liang Yi, Xianrong Long, Qian Pang, Jun Chen, Zheng Wu, Haozhou Shen},
  journal = {Expert Systems}
  title   = {Enhancing Medical Image Segmentation with Feature Fusion Guidance},
  year    = {2026}
}
 ```
