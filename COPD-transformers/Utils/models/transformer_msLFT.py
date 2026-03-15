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

        # Split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=0.1 if self.training else 0
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class DualViewTransformerBlock(nn.Module):
    """
    Processes latents by attending to Axial context, then Coronal context.
    """
    def __init__(self, dim, context_dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.axial_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        self.coronal_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        self.self_attn = CrossAttention(dim, dim, heads, dim_head, dropout) 
        
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x, axial_ctx, coronal_ctx):
        # 1. Attend to Axial Slices
        x = self.axial_attn(x, axial_ctx) + x
        # 2. Attend to Coronal Slices
        x = self.coronal_attn(x, coronal_ctx) + x
        # 3. Latent Self-Communication
        x = self.self_attn(x, x) + x
        # 4. Feed Forward
        x = self.ff(x) + x
        return x

class MultiViewCTTransformer(nn.Module):
    def __init__(self, *, num_slices=20, slice_dim=512, dict_out, 
                 num_latents=16, latent_dim=256, depth=4, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        
        self.dict_out = dict_out
        
        # Linear projections for each view
        self.axial_embed = nn.Sequential(
            nn.Linear(slice_dim, latent_dim),
            nn.LayerNorm(latent_dim))
        
        self.coronal_embed = nn.Sequential(
            nn.Linear(slice_dim, latent_dim),
            nn.LayerNorm(latent_dim))
        
        # Positional encodings
        self.pos_axial = nn.Parameter(torch.randn(1, num_slices, latent_dim) * 0.02)
        self.pos_coronal = nn.Parameter(torch.randn(1, num_slices, latent_dim) * 0.02)
        
        # Learned Latent Queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        self.layers = nn.ModuleList([
            DualViewTransformerBlock(latent_dim, latent_dim, heads, 64, mlp_dim, dropout)
            for _ in range(depth)])

        # Multi-task heads integration
        self.heads = nn.ModuleDict()
        for task, info in dict_out.items():
            # Correctly identify if info is a dimension (int) or list of classes
            if isinstance(info, int):
                out_dim = info
            elif isinstance(info, (list, tuple)):
                out_dim = len(info)
            else:
                out_dim = 1
                
            self.heads[str(task)] = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, out_dim))

    def forward(self, axial_slices, coronal_slices):
        b, s_ax, _ = axial_slices.shape
        _, s_co, _ = coronal_slices.shape
        
        # 1. Embed and add positional info (indexed to actual sequence length)
        ax_ctx = self.axial_embed(axial_slices) + self.pos_axial[:, :s_ax]
        co_ctx = self.coronal_embed(coronal_slices) + self.pos_coronal[:, :s_co]
        
        # 2. Initialize Latents for the batch
        x = repeat(self.latents, '1 n d -> b n d', b=b)
        
        # 3. Iterative Cross-View Attention
        for layer in self.layers:
            x = layer(x, ax_ctx, co_ctx)
        
        # 4. Aggregate latents (Pooling)
        x_pooled = x.mean(dim=1)
        
        # 5. Output dictionary
        return {task: head(x_pooled) for task, head in self.heads.items()}

class transformer_msLFT(nn.Module):
    def __init__(self, *, num_vectors, vec_dim, num_classes, dim, depth, heads, mlp_dim,
                 dim_head=64, dropout=0., emb_dropout=0.):
        
        super().__init__()

        # Wrapper for the Dual-View Fusion
        # We define a dictionary for the output tasks
        # Here we use 'latent' as the key to extract features for the MLP head
        fusion_dict = {"latent": dim} 

        self.fusion_transformer = MultiViewCTTransformer(
            num_slices=num_vectors,
            slice_dim=vec_dim,
            dict_out=fusion_dict,
            num_latents=16, # Or any number of summary tokens
            latent_dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        self.dropout = nn.Dropout(emb_dropout)

        # 2. Final Classification/Regression Head
        # If num_classes > 1: Classification; if num_classes == 1: Regression
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, axial_slices, coronal_slices):
        # MultiViewCTTransformer already handles embedding, pos_encoding, 
        # and latent initialization internally.
        
        # 1. Pass both views into the Cross-Attention Transformer
        # Outputs a dict: {"latent": Tensor([batch, dim])}
        outputs = self.fusion_transformer(axial_slices, coronal_slices)
        
        x = outputs["latent"]
        x = self.dropout(x)
        
        # 2. Final Projection
        return self.mlp_head(x)