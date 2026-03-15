import os
import re
import torch
import numpy as np
import pandas as pd
from torch import nn, einsum
from torch.utils.data import Dataset
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

# --- Helpers ---

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# --- Data Preparation Logic ---

def read_files(file_path, model_id, project_name):
    if model_id == 1 or model_id == 3:
        diagnosis = "finalgold_visit_P1" if model_id == 3 else "emph_cat_P1"
        model_name = "COPDEmph"
    elif model_id == 2:
        diagnosis = "traj"
        model_name = "TRAJ"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

    # Load raw slice-level CSVs
    tr_ax_raw = pd.read_csv(f'{file_path}/training_{model_name}_{project_name}_all.csv', usecols=["File name", diagnosis])
    tr_cor_raw = pd.read_csv(f'{file_path}/training_{model_name}_{project_name}_cor.csv', usecols=["File name", diagnosis])
    va_ax_raw = pd.read_csv(f'{file_path}/validation_{model_name}_{project_name}_all.csv', usecols=["File name", diagnosis])
    va_cor_raw = pd.read_csv(f'{file_path}/validation_{model_name}_{project_name}_cor.csv', usecols=["File name", diagnosis])

    def get_volume_df(df):
        # Take every 20th row to represent the 3D volume
        df_vol = df.iloc[::20].copy()
        # Extract BaseCode (e.g., "18161S") by removing the number and .npy extension
        df_vol['BaseCode'] = df_vol['File name'].apply(lambda x: re.sub(r'\d+\.npy$', '', x))
        return df_vol

    # Process to volume-level and merge on BaseCode to ensure perfect alignment
    tr_ax_v, tr_cor_v = get_volume_df(tr_ax_raw), get_volume_df(tr_cor_raw)
    va_ax_v, va_cor_v = get_volume_df(va_ax_raw), get_volume_df(va_cor_raw)

    train_matched = pd.merge(tr_ax_v, tr_cor_v, on="BaseCode", how="inner", suffixes=('', '_drop'))
    val_matched = pd.merge(va_ax_v, va_cor_v, on="BaseCode", how="inner", suffixes=('', '_drop'))

    return train_matched[['BaseCode', diagnosis]].reset_index(drop=True), \
           train_matched[['BaseCode', diagnosis]].reset_index(drop=True), \
           val_matched[['BaseCode', diagnosis]].reset_index(drop=True), \
           val_matched[['BaseCode', diagnosis]].reset_index(drop=True)

class MyDataset(Dataset):
    def __init__(self, file_list_ax, file_list_cor, add, data_path_ax, data_path_cor, transform=None):
        self.file_list_ax = file_list_ax
        self.file_list_cor = file_list_cor
        self.add = add
        self.path_ax = data_path_ax
        self.path_cor = data_path_cor
        self.transform = transform

    def __len__(self):
        return len(self.file_list_ax)

    def _load_volume(self, data_path, base_code):
        slices = []
        for i in range(1, 21):
            filename = f"{base_code}{i}.npy"
            img_path = os.path.join(data_path, filename)
            
            arr = np.load(img_path).astype(np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0) # 1 x H x W
            
            img = torch.from_numpy(arr)
            if self.transform:
                img = self.transform(img)
            slices.append(img)
        return torch.stack(slices) # Returns (20, 1, H, W)

    def __getitem__(self, idx):
        base_code_ax = self.file_list_ax.iloc[idx, 0]
        base_code_cor = self.file_list_cor.iloc[idx, 0]
        label = self.file_list_ax.iloc[idx, 1] + self.add

        img_ax = self._load_volume(self.path_ax, base_code_ax)
        img_cor = self._load_volume(self.path_cor, base_code_cor)

        return (img_ax, img_cor), label

# --- Transformer and Model Classes ---

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context = None):
        h = self.heads
        x = self.norm(x)
        
        context = default(context, x)
        if context is not x:
            context = self.norm_context(context)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        sim = einsum('b h i d, b h j d -> b h i j', q * self.scale, k)
        attn = sim.softmax(dim = -1)
        
        out = einsum('b h i j, b h j d -> b h i d', self.dropout(attn), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class CrossRegionTransformer(nn.Module):
    def __init__(self, dim, window_size, depth=2, heads=4, dim_head=32):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.window_size = window_size

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head), # Region Self-Attention
                Attention(dim, heads, dim_head), # Cross-View Attention (Ax -> Cor)
                Attention(dim, heads, dim_head), # Cross-View Attention (Cor -> Ax)
                Attention(dim, heads, dim_head), # Local Cross-Attention (Local -> Region)
                FeedForward(dim)
            ]))

    def forward(self, ax_local, ax_region, cor_local, cor_region):
        b, c, lh, lw = ax_local.shape
        _, _, rh, rw = ax_region.shape
        ws_h, ws_w = lh // rh, lw // rw

        ax_l = rearrange(ax_local, 'b c h w -> b (h w) c')
        ax_r = rearrange(ax_region, 'b c h w -> b (h w) c')
        cor_l = rearrange(cor_local, 'b c h w -> b (h w) c')
        cor_r = rearrange(cor_region, 'b c h w -> b (h w) c')

        for r_attn, cross_ax, cross_cor, l_cross, ff in self.layers:
            ax_r = r_attn(ax_r) + ax_r
            cor_r = r_attn(cor_r) + cor_r

            ax_r_fused = cross_ax(ax_r, context = cor_r) + ax_r
            cor_r_fused = cross_cor(cor_r, context = ax_r) + cor_r
            ax_r, cor_r = ax_r_fused, cor_r_fused

            for l_tok, r_tok in [(ax_l, ax_r), (cor_l, cor_r)]:
                w_l = rearrange(l_tok, 'b (h p1 w p2) d -> (b h w) (p1 p2) d', h=rh, w=rw, p1=ws_h, p2=ws_w)
                w_r = rearrange(r_tok, 'b n d -> (b n) 1 d')
                
                w_l = l_cross(w_l, context = w_r) + w_l
                w_l = ff(w_l) + w_l
                
                if l_tok is ax_l: ax_l = rearrange(w_l, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h=rh, w=rw, p1=ws_h)
                else: cor_l = rearrange(w_l, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h=rh, w=rw, p1=ws_h)

        return ax_l, ax_r, cor_l, cor_r

class MultiViewRegionViT(nn.Module):
    def __init__(self, dim=128, num_slices=20, window_size=7, num_classes=2):
        super().__init__()
        self.num_slices = num_slices
        
        self.ax_encoder = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.cor_encoder = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        
        self.region_summarizer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=window_size, p2=window_size),
            nn.Conv2d(dim * (window_size**2), dim, 1))

        self.transformer = CrossRegionTransformer(dim=dim, window_size=window_size)
        
        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, num_classes))

    def forward(self, axial_3d, coronal_3d):
        # axial_3d/coronal_3d: (B, 20, 1, H, W)
        B = axial_3d.shape[0]
        
        ax = rearrange(axial_3d, 'b s c h w -> (b s) c h w')
        cor = rearrange(coronal_3d, 'b s c h w -> (b s) c h w')

        ax_l = self.ax_encoder(ax)
        cor_l = self.cor_encoder(cor)
        
        ax_r = self.region_summarizer(ax_l)
        cor_r = self.region_summarizer(cor_l)

        _, ax_r, _, cor_r = self.transformer(ax_l, ax_r, cor_l, cor_r)

        # Reconstruct spatial dims for region tokens to use the Reduce mean layer
        # If window_size=7 and H=224, encoded L=56, region R=8 (56/7)
        # We need the spatial grid size from region_summarizer output
        h_r = w_r = int(ax_r.shape[1]**0.5)
        ax_r = rearrange(ax_r, 'b (h w) d -> b d h w', h=h_r, w=w_r)

        # Average across the 20 slices
        ax_r = rearrange(ax_r, '(b s) d h w -> b d s h w', b=B, s=self.num_slices)
        ax_r = ax_r.mean(dim=2) # mean over slice dimension
        
        return self.to_logits(ax_r)