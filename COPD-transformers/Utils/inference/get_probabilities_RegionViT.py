# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from __future__ import print_function
import os
import scipy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import vit_pytorch
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm
from torcheval.metrics.functional import multiclass_accuracy
import os
from Utils.models.region_ViT import RegionViT

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
seed = 3
seed_everything(seed)

def PrepareData(file_path,id,batch_size,model_id):
    """
    file_path >> file to call with img file names and labels (dataframe of n_samples x 2)
                 columns = ['File name','Diagnosis']
    manual >> for the particular study, if we use another file_list 
              different than llistat_*_9slices.csv then False, with * number of model (1,2 or 3)
    """
    if model_id == 1 or model_id == 3:
        diagnosis = "finalgold_visit_P1" if model_id == 3 else "emph_cat_P1"
        model_name = "COPDEmph"
    elif model_id == 2:
        diagnosis = "traj"
        model_name = "TRAJ"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    llistat = pd.read_csv(file_path, sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    llistat = llistat.loc[llistat['File name'].str.contains(id)]
    test_pat = llistat.iloc[0:batch_size,:]
    
    return test_pat

def Probabilities(
        main_path,
        model_id,
        batch_size,
        cuda_id,
        id,
        project_name):

        save_path = main_path+project_name+'-results/models'
        data_path = main_path+project_name+'-imgs'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if model_id == 1:
            print('···· EMPHYSEMA MODEL '+str(batch_size)+'-slices ····')
            num_classes = 4
            classes = {0:'No emphysema',1:'Mild emphysema',2:'Moderate emphysema',3:'Extreme emphysema'}
            labels = [0,1,2,3]
            add = 0

        elif model_id == 2:
            print('···· TRAJECTORIES MODEL '+str(batch_size)+'-slices ····')
            num_classes = 6
            classes = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
            labels = [1,2,3,4,5,6]
            add = -1

        elif model_id == 3:
            print('···· COPD MODEL '+str(batch_size)+'-slices ····')
            num_classes = 6
            classes = {0:'COPD -1',1:'COPD 0',2:'COPD 1',3:'COPD 2',4:'COPD 3',5:'COPD 4'}
            labels = [-1,0,1,2,3,4]
            add = 1

        else:
            raise Exception("Sorry, invalid model (must be 1, 2 or 3)")

        cuda_id = cuda_id
        cuda_set = 'cuda:'+str(cuda_id)
        device = torch.device(cuda_set if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        # SETTINGS ----------------------------------------
        # patch_size : int. > Size of patches. image_size must be divisible by patch_size
        patch_size = 4
        # dim : int. > Last dimension of output tensor after linear transformation nn.Linear(..., dim) .
        dim = (128, 256, 512, 1024) # tuple of size 4, indicating dimension at each stage
        # depth : int. > Number of Transformer blocks.
        depth = (2, 2, 14, 2)
        # channels : int, default 3. > Number of image's channels.
        channels = 3
        # dropout : float between [0, 1] , default 0. > Dropout rate.
        dropout = 0.1
        # emb_dropout : float between [0, 1] , default 0 > Embedding dropout rate.
        emb_dropout = 0.1
        window_size = 7

        modelH = RegionViT(
            dim = dim,                             # tuple of size 4, indicating dimension at each stage
            depth = depth,                         # depth of the region to local transformer at each stage
            local_patch_size = patch_size,         # region_patch_size = local_patch_size * window_size = 32 x 7 = 224
            window_size = window_size,             # window size, which should be either 7 or 14
            num_classes = num_classes,             # number of output classes
            tokenize_local_3_conv = False,         # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
            use_peg = True,
            attn_dropout = dropout,
            ff_dropout = emb_dropout,
            channels = channels)

        # -------------------------------------------------------------
        # model type
        model = modelH
        models_path = main_path+project_name+'-checkpoints/checkpoints-RegionViT/'
    
        checkpoint = torch.load(models_path+f"best_{project_name}_{model_id}.pt",map_location='cpu')        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        ################### DATA LOADING ##################

        source_files = main_path+project_name+'-files'
        if model_id == 1 or model_id == 3:
            diagnosis = "finalgold_visit_P1" if model_id == 3 else "emph_cat_P1"
            model_name = "COPDEmph"
            model_name0 = "copd"
        elif model_id == 2:
            diagnosis = "traj"
            model_name0 = "traj"
            model_name = "TRAJ"
        else:
            raise ValueError(f"Unsupported model id: {model_id}")
        if batch_size == 1:
            fusion_type = ''
            file_path = source_files+f'/testing_{model_name}_{project_name}.csv'
        elif batch_size == 9:
            fusion_type = 'C'
            file_path = source_files+f'/testing_{model_name}_{project_name}.csv'
        elif batch_size == 20:
            fusion_type = 'W'
            file_path = source_files+f'/testing_{model_name}_{project_name}_all.csv'
        else:
            raise(Exception('Can only be 1, 9 or 20 slices'))   

        class MyDataset(Dataset):
            def __init__(self, file_list, transform=None):
                self.transform = transform
                self.img_list = file_list.iloc[:,0]
                print("Patient: "+self.img_list.iloc[0])
                if add == 0:
                    self.label_list = file_list.iloc[:,1]
                    print("Emphysema: ",self.label_list.iloc[0])
                elif add == -1:
                    self.label_list = file_list.iloc[:,1]
                    print("Trajectories: ",self.label_list.iloc[0])
                elif add == 1:
                    self.label_list = file_list.iloc[:,1]
                    print("COPD: ",self.label_list.iloc[0])

            def __len__(self):
                self.filelength = len(self.img_list)
                return self.filelength

            def __getitem__(self, idx):
                img_path = "".join([data_path,'/', self.img_list.iloc[idx]])
                label = self.label_list.iloc[idx]+add
                ID = self.img_list.iloc[0]

                if img_path.endswith(".png"):
                    img = Image.open(img_path).convert('RGB')  # 3-channels
                    img = transforms.ToTensor()(img)         # 3 x H x W

                elif img_path.endswith(".npy"):
                    arr = np.load(img_path).astype(np.float32)  # H x W
                    arr = np.expand_dims(arr, axis=0)           # 1 x H x W
                    img = torch.from_numpy(arr)
                    img = img.repeat(3, 1, 1)

                else:
                    raise ValueError("Unsupported file type: " + img_path)
            
                if self.transform:
                    img_transformed = self.transform(img)
                else:
                    img_transformed = img

                return ID, img_transformed, label
        
        applytransforms = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.RandomRotation(20)])
        
        test_pat = PrepareData(file_path,id,batch_size,model_id)
        valid_data = MyDataset(test_pat, transform=applytransforms)
        test_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=False)

        path = f"{main_path}{project_name}-files/df_val_{model_name0}.csv"
        original_labels = pd.read_csv(path,usecols=[diagnosis])
        original_labels = original_labels.dropna().reset_index(drop=True).astype(int)
        original_labels[original_labels == -2.0] = 0.0   # replace -2 with 0

        # unique_values = original_labels[diagnosis].unique()
        # print("Unique values:", unique_values)

        # num_unique = original_labels[diagnosis].nunique()
        # print("Number of unique values:", num_unique)

        test_labels = (original_labels[diagnosis] + add).to_numpy()

        n_classes = num_classes
        test_labels = test_labels.astype(int)

        counts = np.bincount(test_labels, minlength=n_classes)
        proportion = counts/len(test_labels)    
        proportion = torch.tensor(proportion, dtype=torch.float32, device=device)

        real = np.zeros(len(test_labels)//batch_size)
        probable = np.zeros(len(test_labels)//batch_size)
        probabs = np.zeros((len(test_labels)//batch_size,num_classes))
        probabsraw = np.zeros((len(test_labels)//batch_size,num_classes))

        model.eval()
        with torch.no_grad():
            for ix, (ID, data, target) in enumerate(test_loader):
                ID, data, target = ID, data.to(device), target.to(device)

                # --- Model outputs ---
                logits = model(data)                      # [B, C]
                probs = torch.softmax(logits, dim=-1)    # [B, C]

                # --- Class proportion weight vector ---
                weight_vec = 1/proportion
                weight_vec = torch.tensor(weight_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, C]

                if batch_size == 1:
                    # Simple weighting
                    weighted_probs = probs * weight_vec
                    weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)
                    
                    avg_weighted = weighted_probs.mean(dim=0)  # <-- add this
                    pred_class = torch.argmax(weighted_probs, dim=1)
                    real[ix] = target[0].cpu().item()
                    probable[ix] = pred_class.cpu().item()
                    probabs[ix, :] = weighted_probs.cpu().numpy().flatten()
                    probabsraw[ix, :] = probs.cpu().numpy().flatten()

                elif batch_size == 9:
                    # Uniform class weighting + batch averaging
                    weighted_probs = probs * weight_vec
                    weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)

                    avg_weighted = weighted_probs.mean(dim=0)
                    pred_class = torch.argmax(avg_weighted, dim=0)
                    real[ix] = target[0].cpu().item()
                    probable[ix] = pred_class.cpu().item()
                    probabs[ix, :] = avg_weighted.cpu().numpy().flatten()
                    probabsraw[ix, :] = probs.mean(dim=0).cpu().numpy().flatten()

                elif batch_size == 20:
                    # Gaussian batch-position weighting
                    indices = torch.arange(batch_size, device=device)
                    center = (batch_size - 1) / 2
                    width = batch_size / 4
                    mask = (torch.abs(indices - center) < width).float() * 1.0 + \
                        (torch.abs(indices - center) >= width).float() * 0.5
                    mask = mask.unsqueeze(1) / mask.mean()

                    weighted_probs = probs * weight_vec * mask
                    weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)

                    avg_weighted = weighted_probs.mean(dim=0)
                    pred_class = torch.argmax(avg_weighted, dim=0)
                    real[ix] = target[0].cpu().item()
                    probable[ix] = pred_class.cpu().item()
                    probabs[ix, :] = avg_weighted.cpu().numpy().flatten()
                    probabsraw[ix, :] = probs.mean(dim=0).cpu().numpy().flatten()

                ID = str(ID[0])[:-5]
                print(f"Patient {ID} | Probabilities: {probs.mean(dim=0).cpu().tolist()}")
                print(f"Weighted avg: {avg_weighted.cpu().tolist()}")
                print(f"Real: {target[0].cpu().item()-add}, Predicted: {torch.argmax(weighted_probs.mean(dim=0)).cpu().item()-add}")

                # Prepare data for CSV
                data = []
                for idx, prob in enumerate(probs.mean(dim=0).cpu().tolist()):
                    data.append({
                        'class': classes.get(idx, f'class_{idx}'),
                        'probability': prob})

                data_sorted = sorted(data, key=lambda x: x['probability'], reverse=True)
                import csv
                csv_file = project_name+'-results/patients/output_Region_ViT_'+model_name+fusion_type+'_'+ID+'.csv'
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['class', 'probability'])
                    writer.writeheader()
                    writer.writerows(data_sorted)

                print(f"Saved CSV to {csv_file}")
