# LFT-former
## Lung Function Trajectories Transformer Model to predict LFTs from CT images

This repository allows for training and inference of multi-slice LFT to predict lung-function trajectories from CT scans. It allows the user to add clinical features including age, gender, race, smoking-packs-per-year, BMI and % emphysema. This code can be adapted for each project using project_name (e.g. COPDGene in this repo).

### PYTHON DEPENDENCIES
```conda create -n venv python``` <br>
```conda activate venv``` <br>
```pip install -r requirements.txt```

_REQUIRED MODULES:_ 
  - Numpy
  - Pandas
  - Pytorch
  - Pingouin
  - Scikit Learn
  - Tqdm
  - Maplotlib
  - vit-pytorch
  - Pillow
  - Torchvision
  - Torchmetrics
  - Scipy
  - Einops

### FILE ORGANIZATION

`cd your_path`
* git clone into `your_path` -> ```git clone https://github.com/acil-bwh/LFT-former.git```
  - COPD-transformers --> general code files
    > Compute-embeddings <br>
    > Utils <br>
      ...
  - README.md --> how to use this repository
  - requirements.txt --> required libraries
  - folder_setting.py --> generate folders where files, imgs are and where results will be stored

* generate folders -> ```nohup python COPDTransformers/folder_setting.py > folder_setting.log 2>&1 &```
  - .../{project_name}-files --> where files are saved
  - .../{project_name}-imgs --> where imgs in npy or png are saved
  - .../{project_name}-checkpoints --> where checkpoints will be saved
  - .../{project_name}-results --> where your results will be saved
    > figures <br>
    > models <br>
    > patients

<br> 

* a folder {project_name}{M}-features-{N} will be added when computing the embeddings using init_embeddings_RegionViT.py

--> where augmented features of RegionViT are stored (M: model 2 for TRAJ, N:  slices 9 or 20)

**BEFORE RUNNING:** <br>
* activate your virtual environment where python and its libraries are -> ```conda activate venv```

**WHEN RUNNING ON HPC:** <br>
* `nohup` commands: to run a command that keeps running even after you log out/close the terminal/disconnect SSH session <br>

## How to use this repo for training, inference, and visualization

### ARGUMENTS:
  - required:
    - path /your/path/

  - togglable:
    - project_name (default:COPDGene)
    - cuda (default:0)
    - epochs (default:500)
    - batch (default:64)
    - lr (default:1e-6)
    - step (default:10)
    - gamma (default:0.9)
    - decay (default:1e-5)
    - pretrained (default:False call if pretrained)
    - precheck (default:False call if loading from checkpoint)
    - wrapp (default:"modular")
    - add (default:0)
    
<br>

ADD can be: (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) --> Which metadata tensor of clinical features is fused with image-based embeddings:
 
0.	Only CT imaging
1.	CT + Age
2.	CT + Age + Gender
3.	CT + Age + Gender + Race
4.	CT + Age + Gender + Race + %Emphysema (Emph)
5.	CT + Age + Gender + Race + %Emph + BMI
6.	CT + Age + Gender + Race + %Emph + BMI + Packs
7.	CT + Age + BMI
8.	CT + Age + BMI + %Emph
9.	CT + Age + %Emph
10.	CT + %Emph
11.	CT + %Emph + BMI
12.	CT + BMI
13.	CT + Gender
14.	CT + Race
15.	CT + Packs-per-year 

WRAP can be: ("gatt" "gate" "modular" "along" "concat" "dot" "cross" "multi") --> How clinical features are fused with image-based embeddings, default=gatt: gated feature attention [dynamic regulation of the relevance of image-derived features based on the clinical context. We define the query Q as a linear projection of the metadata and the keys K as a linear projection of the feature embeddings. We apply ]

### RegionViT: Training

Running for RegionViT from scratch: 

```python LFT-former/COPD-transformers/init_training_RegionViT.py --path /your/path/ --cuda 1```

Running for RegionViT from a pretrained model:

```python LFT-former/COPD-transformers/init_training_RegionViT.py --path /your/path/ --cuda 1 --pretrained```

Running for RegionViT from a previous checkpoint:

```python LFT-former/COPD-transformers/init_training_RegionViT.py --path /your/path/ --cuda 1 --precheck```

### RegionViT: Obtaining embeddings

```python LFT-former/COPD-transformers/Compute-embeddings/init_embeddings_RegionViT.py --path "/home/qm031/" --cuda 1 --n_augmentations 10```

### LFT-former: Training

Running for LFT-former from scratch: 

```python LFT-former/COPD-transformers/init_training_LFT.py /your/path/ --cuda 1 --add 0 --batch 64 --wrap gatt```

Running for LFT-former from a pretrained model:

```python LFT-former/COPD-transformers/init_training_LFT.py /your/path/ --cuda 1 --add 0 --batch 64 --wrap gatt --pretrained```

Running for LFT-former from a previous checkpoint:

```python LFT-former/COPD-transformers/init_training_LFT.py /your/path/ --cuda 1 --add 0 --batch 64 --wrap gatt --precheck```
  
### LFT-former: Inference

```python LFT-former/COPD-transformers/inference_LFT.py path /your/path/ --cuda 1 --add 0 --wrap gatt```

### LFT-former: Confusion matrices and accuracy results

```python LFT-former/COPD-transformers/visualization_LFT.py path /your/path/ --add 0 --wrap gatt```

### LFT-former: ROC curves

```python LFT-former/COPD-transformers/curves_LFT.py path /your/path/ --add 0 --wrap gatt```

### LFT-former: Predict for a single patient ID

```python LFT-former/COPD-transformers/inference_individual_LFT.py path /your/path/ --cuda 1 --add 0 --wrap gatt```



