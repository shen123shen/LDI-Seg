# LDI-Seg


Many convolutional neural networks have demonstrated outstanding model performance in medical image segmentation task. However, existing networks still exhibit limitations in accurately identifying complex structures. To address this problem, multi-scale feature fusion is revisited in this study. Specifically, how to fuse features from different encoder layers and how to fuse features between encoder and decoder are the focuses. Accordingly, layer-wise feature fusion (LFF) module and decoder semantic enhancement (DSE) module are the solutions respectively. Moreover, a model called LDI-Seg is proposed for medical image segmentation task with the combination of LFF module, DSE module, and pre-trained InceptionNeXt module. To validate the usefulness of the proposed modules, the combination of LFF and DSE are integrated into three popular U-shaped models for testing. Experimental results show that the proposed modules can contribute to model improvement. To verify the effectiveness of the proposed LDI-Seg, four commonly used datasets (ISIC2018, BUSI, Kvasir-SEG and CVC-ClinicDB) are adopted for testing. Results can demonstrate that the LDI-Seg can outperform other popular models in medical image segmentation task.
# Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:\
ISIC-2018 (dermoscopy, 2,594 images fortraining, 100 images for validation, and 1,000 images for testing)\
Kvasir-SEG (gastrointestinal polyp, 600 images for training, 200images for validation, and 200 images for testing)\
BUSI (breast ultrasound, 399 images for training.113 images for validation, and 118 images for testing)\
CVC-ClinicDB (colorectal cancer, 367 images for training, 123images for validation, and 122 images for testing)\
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
  author  = {Haozhou Shen, Shiren Li, Guangguang Yang},
  journal = {Biomedical signal processing and control}
  title   = {A Feature Fusion Guidance for Medical Image Segmentation},
  year    = {2025}
}
 ```
