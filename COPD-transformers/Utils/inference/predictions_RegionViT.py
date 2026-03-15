# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm
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

class MyDataset(Dataset):
    def __init__(self, file_list, add, data_path, transform=None):
        self.img_list = file_list.iloc[:, 0]
        self.label_list = file_list.iloc[:, 1]
        self.add = add
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_list.iloc[idx])
        label = self.label_list.iloc[idx] + self.add

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

        return img_transformed, label
        
def Predictor(main_path,
              model_id,
              batch_size,
              cuda_id,
              project_name):

    if model_id == 1:
        print('···· EMPHYSEMA MODEL ····')
        num_classes = 4
        add = 0
        window_type = "emph"
    elif model_id == 2:
        print('···· TRAJECTORIES MODEL ····')
        num_classes = 6
        add = -1
        window_type = "lung"
    elif model_id == 3:
        print('···· COPD MODEL ····')
        num_classes = 6
        add = +1
        window_type = "copd"
    elif model_id == 4:
        print('···· AIR TRAPPING MODEL ····')
        num_classes = 4
        add = 0
        window_type = "trap"
    elif model_id == 5:
        print('···· COPD 5-class MODEL ····')
        num_classes = 5
        add = 0
        window_type = "copd"
    elif model_id == 6:
        print('···· COPD 2-class MODEL ····')
        num_classes = 2
        add = 0
        window_type = "copd"
    else:
        raise Exception("Sorry, invalid model (must be 1...6)")
    
    source_imgs = main_path+project_name+'-'+window_type
    source_files = main_path+project_name+'-files'
    print("Model:",model_id)

    if model_id == 1 or model_id == 3 or model_id == 4:
        if model_id == 3:
            diagnosis = "finalgold_visit_P1" 
        elif model_id == 4:
            diagnosis = "PRM_pct_airtrapping_Thirona_P1"
        elif model_id == 1:
            diagnosis = "emph_cat_P1"
        model_name = "COPDEmph"
    elif model_id == 2:
        diagnosis = "traj"
        model_name = "TRAJ"
    elif model_id == 5 or model_id == 6: ###### 5 and 2 classes COPD
        diagnosis = "finalgold_visit_P1"
        model_name = "COPDEmph"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    
    if batch_size == 1:
        fusion_type = ''
        test_list = pd.read_csv(source_files+f'/testing_{model_name}_{project_name}.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    elif batch_size == 9:
        fusion_type = 'C'
        test_list = pd.read_csv(source_files+f'/testing_{model_name}_{project_name}.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    elif batch_size == 20:
        fusion_type = 'W'
        test_list = pd.read_csv(source_files+f'/testing_{model_name}_{project_name}_all.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    else:
        raise(Exception('Can only be 1, 9 or 20 slices'))        

    cuda_set = 'cuda:'+str(cuda_id)
    save_path = main_path+project_name+'-results/models'
    data_path = source_imgs
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device(cuda_set if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    path = f"{source_files}/validation_{model_name}_{project_name}.csv"
    original_labels = pd.read_csv(path,usecols=[diagnosis])

    if model_id == 3:
        def bin_labels(val):
            if val < -1:
                return 0
            else: # -1,0,1,2,3,4
                return val
        original_labels = original_labels[diagnosis].apply(bin_labels)

    if model_id == 4:
        def bin_labels(val):
            if val <= 10:
                return 0
            elif 10 < val <= 20:
                return 1
            elif 20 < val <= 40:
                return 2
            else: # > 40
                return 3
        original_labels = original_labels[diagnosis].apply(bin_labels)

    if model_id == 5:
        def bin_labels(val):
            if val <= 0:
                return 0
            else: # 1,2,3,4
                return val
        original_labels = original_labels[diagnosis].apply(bin_labels)

    if model_id == 6:
        def bin_labels(val):
            if val <= 0:
                return 0
            else: # 1,2,3,4
                return 1
        original_labels = original_labels[diagnosis].apply(bin_labels)

    if model_id == 3:
        original_labels[original_labels == -2.0] = 0.0   # replace -2 with 0 if there's some labels = -2 in copd

    original_labels = original_labels.dropna().reset_index(drop=True).astype(int)

    test_labels = (original_labels.squeeze() + add).to_numpy()

    n_classes = num_classes
    test_labels = test_labels.astype(int)
    print(test_labels.shape)

    counts = np.bincount(test_labels, minlength=n_classes)
    num_classes = len(counts)
    total_samples = sum(counts)
    weights = total_samples/(num_classes*counts)
    proportion = torch.tensor(1.0/weights, dtype=torch.float, device=device)

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
    
    checkpoint = torch.load(models_path+f"model_RegionViT_{model_id}.pt",map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device)    

    ################### DATA LOADING ##################
        
    applytransforms = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.RandomRotation(20)])
    
    valid_data = MyDataset(test_list, add, data_path, transform=applytransforms)
    test_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=False)

    real = np.zeros(len(test_list)//batch_size)
    probable = np.zeros(len(test_list)//batch_size)
    probabs = np.zeros((len(test_list)//batch_size,num_classes))
    probabsraw = np.zeros((len(test_list)//batch_size,num_classes))

    model.eval()
    with torch.no_grad():
        for ix, (data, target) in enumerate(tqdm(test_loader, desc="Testing patients")):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            probs = torch.nn.functional.softmax(logits, dim=-1)  # [B, C]
            weight = proportion.to(device)
            weight = weight.unsqueeze(0)                        # [1, C]

            if batch_size == 1:
                weighted_probs = probs * weight
                weighted_probs = weighted_probs/weighted_probs.sum(dim=1, keepdim=True)
                avg_weighted = weighted_probs.mean(dim=0)

                pred_class = torch.argmax(weighted_probs, dim=1)
                real[ix] = target[0].cpu().item() - add
                probable[ix] = pred_class.cpu().item() - add
                probabs[ix, :] = weighted_probs.cpu().numpy().flatten()
                probabsraw[ix, :] = probs.cpu().numpy().flatten()

            elif batch_size == 9:
                weighted_probs = probs * weight
                weighted_probs = weighted_probs/weighted_probs.sum(dim=1, keepdim=True)
                avg_weighted = weighted_probs.mean(dim=0)
                avg_probs = probs.mean(dim=0)
                pred_class = torch.argmax(avg_probs, dim=0)
                real[ix] = target[0].cpu().item() - add
                probable[ix] = pred_class.cpu().item() - add
                probabs[ix, :] = avg_weighted.cpu().numpy().flatten()
                probabsraw[ix, :] = probs.mean(dim=0).cpu().numpy().flatten()

            elif batch_size == 20:
                indices = torch.arange(batch_size, device=device)
                weighting = "piecewise"
                if weighting == "gaussian":
                    # Gaussian batch-position weighting
                    indices = torch.arange(batch_size, device=device)
                    center = (batch_size - 1) / 2
                    sigma = batch_size / num_classes
                    mask = torch.exp(-0.5 * ((indices - center) / sigma) ** 2)
                    mask = mask / mask.max()
                    mask = mask.unsqueeze(1) / mask.mean()  # [B,1]

                elif weighting == "exponential":
                    indices = torch.arange(batch_size, device=device)
                    center = (batch_size - 1) / 2
                    decay = 2.0  # higher = sharper focus
                    mask = torch.exp(-decay * torch.abs(indices - center) / batch_size)
                    mask = mask.unsqueeze(1) / mask.mean()

                elif weighting == "cosine":
                    import math
                    indices = torch.arange(batch_size, device=device)
                    mask = 0.5 * (1 + torch.cos(math.pi * (indices - (batch_size - 1) / 2) / (batch_size / 2)))
                    mask = mask.unsqueeze(1) / mask.mean()

                elif weighting == "piecewise":
                    indices = torch.arange(batch_size, device=device)
                    center = (batch_size - 1) / 2
                    width = batch_size / 4
                    mask = (torch.abs(indices - center) < width).float() * 1.0 + \
                        (torch.abs(indices - center) >= width).float() * 0.5
                    mask = mask.unsqueeze(1) / mask.mean()

                else:
                    raise(Exception('Weighting type is not accepted'))

                weighted_probs = probs * weight  # class weighting
                weighted_probs = weighted_probs * mask  # position weighting
                weighted_probs = weighted_probs/weighted_probs.sum(dim=1, keepdim=True)
                avg_weighted = weighted_probs.mean(dim=0)
                avg_probs = probs.mean(dim=0)
                pred_class = torch.argmax(avg_probs, dim=0)
                real[ix] = target[0].cpu().item() - add
                probable[ix] = pred_class.cpu().item() - add
                probabs[ix, :] = avg_weighted.cpu().numpy().flatten()
                probabsraw[ix, :] = probs.mean(dim=0).cpu().numpy().flatten()

    df = pd.DataFrame(columns=['data','label','predicted'])
    df['data'] = test_list.iloc[::batch_size,0]
    df['label'] = test_list.iloc[::batch_size,1]
    df['predicted'] = probable.T
    df.to_csv(save_path+"/model_RegionViT_"+str(model_id)+"_"+fusion_type+".csv", index=False)

    if model_id == 1 or model_id == 4:
        df = pd.DataFrame(columns=['data','label','probs0','probs1','probs2','probs3'])
        df['data'] = test_list.iloc[::batch_size,0]
        df['label'] = test_list.iloc[::batch_size,1]
        df['probs0'] = probabs[:,0]
        df['probs1'] = probabs[:,1]
        df['probs2'] = probabs[:,2]
        df['probs3'] = probabs[:,3]
        df.to_csv(save_path+"/probs_RegionViT_"+str(model_id)+"_"+fusion_type+".csv", index=False)

    elif model_id == 2:
        df = pd.DataFrame(columns=['data','label','probs1','probs2','probs3','probs4','probs5','probs6'])
        df['data'] = test_list.iloc[::batch_size,0]
        df['label'] = test_list.iloc[::batch_size,1]
        df['probs1'] = probabs[:,0]
        df['probs2'] = probabs[:,1]
        df['probs3'] = probabs[:,2]
        df['probs4'] = probabs[:,3]
        df['probs5'] = probabs[:,4]
        df['probs6'] = probabs[:,5]
        df.to_csv(save_path+"/probs_RegionViT_"+str(model_id)+"_"+fusion_type+".csv", index=False)

    elif model_id == 3:
        df = pd.DataFrame(columns=['data','label','probs-1','probs0','probs1','probs2','probs3','probs4'])
        df['data'] = test_list.iloc[::batch_size,0]
        df['label'] = test_list.iloc[::batch_size,1]
        df['probs-1'] = probabs[:,0]
        df['probs0'] = probabs[:,1]
        df['probs1'] = probabs[:,2]
        df['probs2'] = probabs[:,3]
        df['probs3'] = probabs[:,4]
        df['probs4'] = probabs[:,5]
        df.to_csv(save_path+"/probs_RegionViT_"+str(model_id)+"_"+fusion_type+".csv", index=False)

    elif model_id == 5:
        df = pd.DataFrame(columns=['data','label','probs0','probs1','probs2','probs3','probs4'])
        df['data'] = test_list.iloc[::batch_size,0]
        df['label'] = test_list.iloc[::batch_size,1]
        df['probs0'] = probabs[:,0]
        df['probs1'] = probabs[:,1]
        df['probs2'] = probabs[:,2]
        df['probs3'] = probabs[:,3]
        df['probs4'] = probabs[:,4]
        df.to_csv(save_path+"/probs_RegionViT_"+str(model_id)+"_"+fusion_type+".csv", index=False)

    elif model_id == 6:
        df = pd.DataFrame(columns=['data','label','probs0','probs1'])
        df['data'] = test_list.iloc[::batch_size,0]
        df['label'] = test_list.iloc[::batch_size,1]
        df['probs0'] = probabs[:,0]
        df['probs1'] = probabs[:,1]
        df.to_csv(save_path+"/probs_RegionViT_"+str(model_id)+"_"+fusion_type+".csv", index=False)
        
    print('FINISHED PREDICTING VALUES AND PROBABILITIES')
    return