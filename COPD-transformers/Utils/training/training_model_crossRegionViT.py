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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm
from torcheval.metrics.functional import multiclass_accuracy
import os
from Utils.models.crossregion_ViT import MultiViewRegionViT
import warnings
warnings.filterwarnings("ignore")
import re

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

def read_files(file_path,model_id,project_name):
    if model_id == 1 or model_id == 3:
        diagnosis = "finalgold_visit_P1" if model_id == 3 else "emph_cat_P1"
        model_name = "COPDEmph"
    elif model_id == 2:
        diagnosis = "traj"
        model_name = "TRAJ"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    
    train_list = pd.read_csv(file_path+f'/training_{model_name}_{project_name}_all.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)
    val_list = pd.read_csv(file_path+f'/validation_{model_name}_{project_name}_all.csv', sep=",", usecols=['File name',diagnosis],header=0,low_memory=False)

    def get_volume_df(df):
        df_vol = df.iloc[::20].copy()
        df_vol['BaseCode'] = df_vol['File name'].apply(lambda x: re.sub(r'\d+\.npy$', '', x))
        return df_vol

    train_list = get_volume_df(train_list)
    val_list = get_volume_df(val_list)

    train_list = train_list[['BaseCode', diagnosis]].reset_index(drop=True)
    val_list = val_list[['BaseCode', diagnosis]].reset_index(drop=True)

    return train_list, val_list

class MyDataset(Dataset):
    def __init__(self, file_list, add, data_path_ax, data_path_cor, transform=None):
        self.file_list = file_list
        self.add = add
        self.path_ax = data_path_ax
        self.path_cor = data_path_cor
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def _load_volume(self, data_path, base_code):
        slices = []
        for i in range(1, 21):
            filename = f"{base_code}{i}.npy"
            img_path = os.path.join(data_path, filename)
            
            arr = np.load(img_path).astype(np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0) 
            
            img = torch.from_numpy(arr)
            
            # repeat to 3 channels if necessary
            # img = img.repeat(3, 1, 1)

            if self.transform:
                img = self.transform(img)
            
            slices.append(img)
            
        return torch.stack(slices)

    def __getitem__(self, idx):
        base_code = self.file_list.iloc[idx, 0]
        label = self.file_list.iloc[idx, 1] + self.add
        img_ax = self._load_volume(self.path_ax, base_code)
        img_cor = self._load_volume(self.path_cor, base_code)

        return img_ax, img_cor, label

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
        step):

        source_imgs_ax = project_name+'-imgs'
        source_imgs_cor = project_name+'-imgs-cor'
        source_files = project_name+'-files'

        train_list, valid_list = read_files(source_files,model_id,project_name)

        cuda_id = int(cuda_id)
        cuda_set = 'cuda:'+str(cuda_id)
        main_checkpoint_path = main_path+project_name+'-checkpoints/checkpoints-crossRegionViT'

        data_path_ax = main_path+source_imgs_ax+'/'
        data_path_cor = main_path+source_imgs_cor+'/'

        if not os.path.exists(main_checkpoint_path):
            os.makedirs(main_checkpoint_path)

        device = torch.device(cuda_set if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        if model_id == 1:
            print('···· EMPHYSEMA MODEL ····')
            num_classes = 4
            add = 0
        elif model_id == 2:
            print('···· TRAJECTORIES MODEL ····')
            num_classes = 6
            add = -1
        elif model_id == 3:
            print('···· COPD MODEL ····')
            num_classes = 6
            add = +1
        else:
            raise Exception("Sorry, invalid model (must be 1, 2 or 3)")
        
        # SETTINGS ----------------------------------------
        image_size = 224
        channels = 1
        num_slices = 20
        dropout = 0.1
        dim = 128
        window_size = 7

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
        print(f"Window: {window_size}")
        print(f"Dropout: {dropout}")
        print(f"Number of slices: {num_slices}")
        print(f"Dimension: {dim}")
        print(" ")

        model = MultiViewRegionViT(dim=dim, num_slices=num_slices, window_size=window_size, num_classes=num_classes).to(device=device)
        
        # -------------------------------------------------------------
        
        # optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
        
        # scheduler
        scheduler = StepLR(optimizer, step_size=step, gamma=gamma)
        
        if load_pretrained == True and load_from_checkpoint == False:
            checkpoint_path = os.path.join(main_checkpoint_path, f"best_{project_name}_{model_id}_crossRegionViT.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device=device)
            last_epoch = 0
            best_val_acc = 0.0
            print(f"Loaded pretrained weights from {checkpoint_path}\n")

        elif load_from_checkpoint == True and load_pretrained == False:
            checkpoint_path = os.path.join(main_checkpoint_path, f"best_{project_name}_{model_id}_crossRegionViT.pt")
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
        print("Model: crossRegionViT")
        print("Loss: Cross-Entropy")
        print("Optimizer: Adam")
        print("Scheduler: StepLR")
        print(" ")

        ################### DATA LOADING ##################
        applytransforms = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.RandomRotation(20)])

        train_data = MyDataset(train_list, add, data_path_ax, data_path_cor, applytransforms)
        valid_data = MyDataset(valid_list, add, data_path_ax, data_path_cor, applytransforms)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

        #################### TRAIN/VAL ####################
        training_loss, validation_loss = [], []
        training_acc, validation_acc = [], []

        print(f"Training for {epochs} epochs...\n")

        best_epoch = last_epoch
        epochs_no_improve = 0
        patience = step*2
        best_val_loss = float('inf')

        # loss function
        criterion = nn.CrossEntropyLoss()

        for epoch in range(last_epoch, last_epoch + epochs):
            model.train()
            total_loss, total_acc = 0, 0

            for axial, coronal, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                axial = axial.to(device)
                coronal = coronal.to(device)
                labels = labels.to(device)
                labels[labels == -1.0] = 0.0   # replace -1 with 0
                labels = labels.long().to(device)
                
                optimizer.zero_grad()

                outputs = model(axial, coronal)
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
                for axial, coronal, labels  in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                    axial = axial.to(device)
                    coronal = coronal.to(device)
                    labels = labels.to(device)
                    labels[labels == -1.0] = 0.0   # replace -1 with 0
                    labels = labels.long().to(device)

                    outputs = model(axial, coronal)

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

                best_model_path = os.path.join(main_checkpoint_path, f"best_{project_name}_{model_id}_crossRegionViT.pt")
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
            
        final_model_path = os.path.join(main_path, f"{project_name}-models/model_{project_name}_{model_id}_crossRegionViT.pt")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": validation_loss[-1]}, final_model_path)

        print("\nTraining complete.")
        print(f"Saved final model at: {final_model_path}")
        print(f"Best model saved at epoch {last_epoch+best_epoch} with val_acc: {best_val_acc:.4f}")

        return {
            "train_loss": training_loss,
            "val_loss": validation_loss,
            "train_acc": training_acc,
            "val_acc": validation_acc}