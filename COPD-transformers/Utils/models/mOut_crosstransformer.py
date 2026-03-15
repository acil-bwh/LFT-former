import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x, context):
        x = self.norm(x)
        context = self.norm_context(context)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LatentTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        self.self_attn = CrossAttention(dim, dim, heads, dim_head, dropout) 
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        x = self.cross_attn(x, context) + x
        x = self.self_attn(x, x) + x
        x = self.ff(x) + x
        return x

class MultiOutputDiagnosticTransformer(nn.Module):
    def __init__(self, *, num_slices, slice_dim, dict_out, num_latents=16, 
                 latent_dim=256, depth=4, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        
        self.dict_out = dict_out
        
        self.slice_embed = nn.Sequential(
            nn.Linear(slice_dim, latent_dim),
            nn.LayerNorm(latent_dim))
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_slices, latent_dim) * 0.02)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        self.layers = nn.ModuleList([
            LatentTransformerBlock(latent_dim, latent_dim, heads, 64, mlp_dim, dropout)
            for _ in range(depth)])

        self.heads = nn.ModuleDict()
        for task, info in dict_out.items():
            out_dim = len(info) if isinstance(info, (list, tuple)) else 1
            self.heads[f"task{task}"] = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, out_dim))

    def forward(self, slices):
        b, s, d = slices.shape
        
        context = self.slice_embed(slices)
        context += self.pos_embedding[:, :s]        
        
        x = repeat(self.latents, '1 n d -> b n d', b=b)
        
        for layer in self.layers:
            x = layer(x, context)
        
        x = x.mean(dim=1)
        
        out = {}
        for task, head in self.heads.items():
            out[task] = head(x)

        return out