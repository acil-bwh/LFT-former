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
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm
from torcheval.metrics.functional import multiclass_accuracy
import os
from Utils.models.region_ViT import RegionViT
import warnings
warnings.filterwarnings("ignore")


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

def read_files(file_path,model_id,project_name,axis):
    if model_id == 1 or model_id == 3:
        diagnosis = "finalgold_visit_P1" if model_id == 3 else "emph_cat_P1"
        model_name = "COPDEmph"
    elif model_id == 2:
        diagnosis = "traj"
        model_name = "TRAJ"
    elif model_id == 4:
        diagnosis = "PRM_pct_airtrapping_Thirona_P1"
        model_name = "COPDEmph"
    elif model_id == 5 or model_id == 6: ###### 5 and 2 classes COPD
        diagnosis = "finalgold_visit_P1"
        model_name = "COPDEmph"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    
    if axis == "coronal":
        axis_suffix = "_cor"
    elif axis == "axial":
        axis_suffix = ""

    train_list = pd.read_csv(file_path+f'/training_{model_name}_{project_name}{axis_suffix}.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    val_list = pd.read_csv(file_path+f'/validation_{model_name}_{project_name}{axis_suffix}.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    
    if model_id == 3:
        def bin_labels(val):
            if val < -1: # -2
                return 0
            else: # -1,0,1,2,3,4
                return val

        train_list[diagnosis] = train_list[diagnosis].apply(bin_labels)
        val_list[diagnosis] = val_list[diagnosis].apply(bin_labels)

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

        train_list[diagnosis] = train_list[diagnosis].apply(bin_labels)
        val_list[diagnosis] = val_list[diagnosis].apply(bin_labels)

    if model_id == 5:
        def bin_labels(val):
            if val <= 0:
                return 0
            else: # 1,2,3,4
                return val

        train_list[diagnosis] = train_list[diagnosis].apply(bin_labels)
        val_list[diagnosis] = val_list[diagnosis].apply(bin_labels)
    
    if model_id == 6:
        def bin_labels(val):
            if val <= 0:
                return 0
            else: # 1,2,3,4
                return 1

        train_list[diagnosis] = train_list[diagnosis].apply(bin_labels)
        val_list[diagnosis] = val_list[diagnosis].apply(bin_labels)

    train_list = train_list[["File name", diagnosis]]
    val_list = val_list[["File name", diagnosis]]
    return train_list, val_list

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

def Trainer(main_path,
        model_id,
        batch_size,
        epochs,
        cuda_id,
        load_from_checkpoint,
        load_pretrained,
        project_name,
        lr,
        gamma,
        weight_decay,
        step,
        axis):
        
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
        
        if axis == "coronal":
            axis_suffix = "_cor"
            model_suffix = "-cor"
        elif axis == "axial":
            source_imgs = f"{project_name}-{window_type}"
            axis_suffix = ""
            model_suffix = ""

        source_files = project_name+'-files'
        source_imgs = f"{project_name}-{window_type}{model_suffix}"

        train_list, valid_list = read_files(source_files,model_id,project_name,axis)

        cuda_id = int(cuda_id)
        cuda_set = 'cuda:'+str(cuda_id)
        main_checkpoint_path = main_path+project_name+f'-checkpoints/checkpoints-RegionViT{model_suffix}'
        data_path = main_path+source_imgs+'/'

        if not os.path.exists(main_checkpoint_path):
            os.makedirs(main_checkpoint_path)

        device = torch.device(cuda_set if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        class_counts = train_list.iloc[:,1].value_counts().sort_index().values
        num_classes = len(class_counts)
        total_samples = sum(class_counts)
        weights = total_samples/(num_classes*class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)

        print(f"Counts per bin: {class_counts}")
        print(f"Weights per bin: {class_weights}")
        
        # SETTINGS ----------------------------------------
        # image_size : int. > Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
        image_size = 224
        # patch_size : int. > Size of patches. image_size must be divisible by patch_size
        patch_size = 4
        # The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16.
        num_patches = int(image_size*image_size/patch_size**2)
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
        region_patch = patch_size*window_size

        print("--- Training characteristics ---")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Gamma: {gamma}")
        print(f"Device: {device}")
        print(" ")
        print("--- Input characteristics ---")
        print(f"Image (cxhxw): {channels}x{image_size}x{image_size}")
        print(f"Label (#): {num_classes}")
        print(" ")
        print("--- Model characteristics ---")
        print(f"Local patch size: {patch_size}")
        print(f"Number of local patches: {num_patches}")
        print(f"Window: {window_size}")
        print(f"Region patch size: {region_patch}")
        print(f"Dropout: {dropout}")
        print(f"Embedding dropout: {emb_dropout}")
        print(f"Depth: {depth}")
        print(f"Dimension: {dim}")
        print(f"Axis: {axis}")
        print(" ")

        model = RegionViT(
            dim = dim,                             # tuple of size 4, indicating dimension at each stage
            depth = depth,                         # depth of the region to local transformer at each stage
            local_patch_size = patch_size,         # region_patch_size = local_patch_size * window_size = 32 x 7 = 224
            window_size = window_size,             # window size, which should be either 7 or 14
            num_classes = num_classes,             # number of output classes
            tokenize_local_3_conv = False,         # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
            use_peg = True,
            attn_dropout = dropout,
            ff_dropout = emb_dropout,
            channels = channels).to(device=device)
        
        # -------------------------------------------------------------
        
        # optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
        
        # scheduler
        scheduler = StepLR(optimizer, step_size=step, gamma=gamma)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        if load_pretrained == True and load_from_checkpoint == False:
            checkpoint_path = os.path.join(main_checkpoint_path, f"best_RegionViT_{model_id}{axis_suffix}.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device=device)
            last_epoch = 0
            best_val_acc = 0.0
            print(f"Loaded pretrained weights from {checkpoint_path}\n")

        elif load_from_checkpoint == True and load_pretrained == False:
            checkpoint_path = os.path.join(main_checkpoint_path, f"best_RegionViT_{model_id}{axis_suffix}.pt")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device=device)
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            best_val_acc = checkpoint.get('val_acc', 0.0)  # load the best validation accuracy
            last_epoch = checkpoint.get('epoch', 0)
            print(f"Loaded best model from epoch {last_epoch}")

        elif load_from_checkpoint == False and load_pretrained == False:
            model = model.to(device=device)
            last_epoch = 0
            best_val_acc = 0.0
            print("Starting training from scratch")

        else:
            print(f"load_pretrained is set to: {load_pretrained}")
            print(f"load_from_checkpoint is set to: {load_from_checkpoint}")
            raise ValueError("You cannot set both 'load_pretrained' and 'load_from_checkpoint' to the same value.")

        ######## change for own choices IF MODIFIED ABOVE ########
        print("Model: RegionViT")
        print("Loss: Cross-Entropy")
        print("Optimizer: Adam")
        print("Scheduler: StepLR")
        print(" ")

        ################### DATA LOADING ##################
        applytransforms = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.RandomRotation(20)])

        train_data = MyDataset(train_list, add, data_path, applytransforms)
        valid_data = MyDataset(valid_list, add, data_path, applytransforms)

        train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)

        #################### TRAIN/VAL ####################
        training_loss, validation_loss = [], []
        training_acc, validation_acc = [], []

        print(f"Training for {epochs} epochs...\n")

        best_epoch = last_epoch
        epochs_no_improve = 0
        patience = step*2
        best_val_loss = float('inf')

        # loss function
        criterion = nn.CrossEntropyLoss(class_weights)

        for epoch in range(last_epoch, last_epoch + epochs):
            model.train()
            total_loss, total_acc = 0, 0

            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                data, labels = data.to(device), labels.to(device)

                if model_id == 3:
                    labels[labels == -1.0] = 0.0   # replace -1 with 0

                labels = labels.long().to(device)
                
                optimizer.zero_grad()

                outputs = model(data)
                loss = criterion(outputs, labels)
                acc = multiclass_accuracy(outputs, labels, num_classes=num_classes)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += acc.item()

            avg_train_loss = total_loss / len(train_loader)
            avg_train_acc = total_acc / len(train_loader)
            training_loss.append(avg_train_loss)
            training_acc.append(avg_train_acc)

            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for data, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                    data, labels = data.to(device), labels.to(device)

                    if model_id == 3:
                        labels[labels == -1.0] = 0.0   # replace -1 with 0

                    labels = labels.long().to(device)

                    outputs = model(data)

                    loss = criterion(outputs, labels)
                    acc = multiclass_accuracy(outputs, labels, num_classes=num_classes)

                    val_loss += loss.item()
                    val_acc += acc.item()

            avg_val_loss = val_loss / len(valid_loader)
            avg_val_acc = val_acc / len(valid_loader)
            validation_loss.append(avg_val_loss)
            validation_acc.append(avg_val_acc)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

            print(f"Epoch {epoch+1}: Train loss={avg_train_loss:.4f}, acc={avg_train_acc:.4f} | "
                f"Val loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}")

            improved_acc = avg_val_acc > best_val_acc + 1e-4
            improved_loss = avg_val_loss < best_val_loss - 1e-4

            if improved_acc or improved_loss:
                best_val_acc = max(best_val_acc, avg_val_acc)
                best_val_loss = min(best_val_loss, avg_val_loss)
                best_epoch = epoch + 1
                epochs_no_improve = 0

                best_model_path = os.path.join(main_checkpoint_path, f"best_RegionViT_{model_id}{axis_suffix}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                    "val_acc": best_val_acc}, best_model_path)
                print(f"Best model updated at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")
            
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} (val_acc: {best_val_acc:.4f})")
                break
            
        final_model_path = os.path.join(main_path, f"{project_name}-models/model_RegionViT_{model_id}{axis_suffix}.pt")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": validation_loss[-1],
        }, final_model_path)

        print("\nTraining complete.")
        print(f"Saved final model at: {final_model_path}")
        print(f"Best model saved at epoch {last_epoch+best_epoch} with val_acc: {best_val_acc:.4f}")

        return {
            "train_loss": training_loss,
            "val_loss": validation_loss,
            "train_acc": training_acc,
            "val_acc": validation_acc}