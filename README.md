# LDI-Seg


Medical image segmentation is pivotal in clinical applications such as disease diagnosis and treatment planning. Despite advancements, accurately identifying complex anatomical structures remains challenging. This study introduces a Layer-wise Feature Fusion (LFF) module and a Decoder Semantic Enhancement (DSE) module to enhance feature representation and segmentation accuracy. The proposed LDI-Seg model integrates these modules with a pre-trained InceptionNeXt encoder. Experimental results on four public datasets demonstrate that LDI-Seg outperforms existing models, achieving state-of-the-art accuracy without excessive computational overhead. Here, we show that our approach significantly improves lesion recognition capability, offering a promising solution for medical image segmentation tasks. The source code is available at: https://github.com/shen123shen/LDI-Seg.
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
  journal = {The Vision Computer}
  title   = {Feature Fusion-Enhanced Medical Image Segmentation: A Novel Approach},
  year    = {2025}
}
 ```
