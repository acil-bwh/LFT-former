import torch
from torch import nn
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class mOut_transformer(nn.Module):
    def __init__(self, *, num_vectors, vec_dim, dict_out, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, "pool type must be 'cls' or 'mean'"

        self.dict_out = dict_out
        self.pool = pool

        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(vec_dim),
            nn.Linear(vec_dim, dim),
            nn.LayerNorm(dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_vectors + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.heads = nn.ModuleDict()
        for task, values in dict_out.items():
            if isinstance(values, list):
                # Classification head
                self.heads[f"task{task}"] = nn.Linear(dim, len(values))
            else:
                # Regression head
                self.heads[f"task{task}"] = nn.Linear(dim, 1)

    def forward(self, x):
        b, n, _ = x.shape

        x = self.patch_to_embedding(x)

        # Assuming n == self.num_vectors, no interpolation is needed
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding # Shape (1, n+1, d) added to (b, n+1, d)
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        out = {}
        for task, head in self.heads.items():
            out[task] = head(x)
        return out