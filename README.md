# LFT-former
## Multi-slice transformer model to predict lung-function trajectories (LFTs) from CT images

This repository allows for training and inference of multi-slice LFT to predict lung-function trajectories from CT scans. It allows the user to add clinical features including age, gender, race, smoking-packs-per-year, BMI and % emphysema. This code can be adapted for each project using project_name.

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
  - Functions --> general code files
    > Utils <br>
      ...
  - README.md --> how to use this repository
  - requirements.txt --> required libraries
  - folder_setting.py --> generate folders where files, imgs are and where results will be stored
  - file_processing.py --> generate files for training, validation and testing (requires main path, project name and total number of slices [in this example 9 for central and 20 for total, but can be toggled to your own project and CT subsampling])

* generate folders -> ```python LFT-former/folder_setting.py``` 

__OVERALL HIERARCHY BEFORE RUNNING LFT:__

  - .../{project_name}-files --> store here your source csv files (see A)
  - .../{project_name}-imgs --> store here your imgs (see B)
  - .../{project_name}-checkpoints --> where checkpoints will be saved
  - .../{project_name}-models --> where final models will be saved
  - .../{project_name}-embeddings --> where embeddings will be saved
  - .../{project_name}-results --> where your results will be saved
    > figures <br>
    > models <br>
    > metrics

IMAGE MANAGEMENT:
Store the 2D scans in npy or png for IDs=1...N, in our example slices=20
  > ID1_1.npy <br>
  ... <br>
  > ID1_20.npy <br>
  ... <br>
  > IDN_1.npy <br>
  ... <br>
  > IDN_20.npy <br>

FILE MANAGEMENT:
In your original file, variables should be stored as: ["sid", "traj", "age", "gender", "packs", "emph", "race", "bmi"] and should be stored in {project_name}-files. Then, create the necessary subfiles using the provided function -> ```python LFT-former/file_processing.py```

## USAGE MANUAL

**BEFORE RUNNING:** <br>
* activate your virtual environment where python and its libraries are -> ```conda activate venv```

**WHEN RUNNING ON HPC:** <br>
* `nohup` commands: to run a command that keeps running even after you log out/close the terminal/disconnect SSH session <br>

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
    - wrapp (default:"gatt")
    - add (default:0)
    
ADD can be: (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) --> Which metadata tensor of clinical features is fused with image-based embeddings. 

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

## CALLING AND EXECUTING
### RegionViT: Training

Running for RegionViT from scratch: 

```python LFT-former/Functions/init_training_RegionViT.py --path /your/path/ --cuda 1```

Running for RegionViT from a pretrained model:

```python LFT-former/Functions/init_training_RegionViT.py --path /your/path/ --cuda 1 --pretrained```

Running for RegionViT from a previous checkpoint:

```python LFT-former/Functions/init_training_RegionViT.py --path /your/path/ --cuda 1 --precheck```

### RegionViT: Obtaining embeddings

```python LFT-former/Functions/init_embeddings_RegionViT.py --path "/home/qm031/" --cuda 1 --n_augmentations 10```

### LFT-former: Training

Running for LFT-former from scratch: 

```python LFT-former/Functions/init_training_LFT.py /your/path/ --cuda 1 --add 0 --batch 64 --wrap gatt```

Running for LFT-former from a pretrained model:

```python LFT-former/Functions/init_training_LFT.py /your/path/ --cuda 1 --add 0 --batch 64 --wrap gatt --pretrained```

Running for LFT-former from a previous checkpoint:

```python LFT-former/Functions/init_training_LFT.py /your/path/ --cuda 1 --add 0 --batch 64 --wrap gatt --precheck```
  
### LFT-former: Inference

```python LFT-former/Functions/inference_LFT.py path /your/path/ --cuda 1 --add 0 --wrap gatt```

### LFT-former: Confusion matrices and accuracy results

```python LFT-former/Functions/visualization_LFT.py path /your/path/ --add 0 --wrap gatt```

### LFT-former: ROC curves

```python LFT-former/Functions/curves_LFT.py path /your/path/ --add 0 --wrap gatt```

### LFT-former: Predict for a single patient ID

```python LFT-former/Functions/inference_individual_LFT.py path /your/path/ --cuda 1 --add 0 --wrap gatt```



