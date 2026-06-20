import math, os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# ── Model Architectures ───────────────────────────────────────────────────────

class PosEnc(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PolyGuidedFusion(nn.Module):
    def __init__(self, sk_dim=96, po_dim=64, vis_dim=192, d=256,
                 h=4, L=3, nc=2, dr=0.1):
        super().__init__()
        self.sk_proj  = nn.Sequential(nn.Linear(sk_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dr))
        self.po_proj  = nn.Sequential(nn.Linear(po_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dr))
        self.vis_proj = nn.Sequential(nn.Linear(vis_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dr))

        self.cross_attn = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.cross_ff   = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ff_norm    = nn.LayerNorm(d)

        self.sk_mod   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.vis_mod  = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos_enc  = PosEnc(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=4*d,
            dropout=dr, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=L)
        self.drop = nn.Dropout(dr)
        self.head = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Dropout(dr), nn.Linear(d//2, nc))

    def forward(self, sk, po, vi):
        B = sk.size(0)
        s = self.sk_proj(sk)
        p = self.po_proj(po)
        v = self.vis_proj(vi)
        attn, _ = self.cross_attn(query=p, key=s, value=s)
        p2 = self.cross_norm(p + attn)
        p2 = self.ff_norm(p2 + self.cross_ff(p2))
        p2 = p2 + self.sk_mod
        v = v + self.vis_mod
        cls = self.cls_token.expand(B, -1, -1)
        x = self.pos_enc(torch.cat([cls, p2, v], dim=1))
        z = self.drop(self.encoder(x)[:, 0])
        return self.head(z), z


class StreamingLDA:
    def __init__(self, input_dim, num_classes, shrinkage=1e-4, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.shrinkage = shrinkage
        self.device = device
        self.muK = torch.zeros(num_classes, input_dim).to(device)
        self.cK = torch.zeros(num_classes).to(device)
        self.Sigma = torch.eye(input_dim).to(device)
        self.num_updates = 0
        self._Lambda = None

    def _invalidate(self):
        self._Lambda = None

    @property
    def Lambda(self):
        if self._Lambda is None:
            reg = (1 - self.shrinkage) * self.Sigma + \
                  self.shrinkage * torch.eye(self.input_dim).to(self.device)
            self._Lambda = torch.linalg.pinv(reg)
        return self._Lambda

    def predict_proba(self, X, batch=1024):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.float().to(self.device)
        N = X.shape[0]
        Lam = self.Lambda
        M = self.muK.T
        W = Lam @ M
        c = 0.5 * (M * W).sum(0)
        scores = torch.empty(N, self.num_classes, device=self.device)
        for s in range(0, N, batch):
            e = min(s + batch, N)
            scores[s:e] = X[s:e] @ W - c
        return torch.softmax(scores, 1).cpu().numpy().astype(np.float32)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.muK = ck['muK'].to(self.device)
        self.cK = ck['cK'].to(self.device)
        self.Sigma = ck['Sigma'].to(self.device)
        self.num_updates = ck['num_updates']
        self._invalidate()
        print(f'[SLDA] Loaded <- {path}')


def load_polyflow_twostage(
    fusion_det_ckpt: str,
    slda_binary_ckpt: str,
    fusion_rec_ckpt: Optional[str] = None,
    slda_14class_ckpt: Optional[str] = None,
    device: str = "cuda"
):
    """Factory function to load the full model stack."""
    dev = device if torch.cuda.is_available() else "cpu"
    
    # 1. Anomaly Detection
    det_model = PolyGuidedFusion(nc=2).to(dev)
    if os.path.exists(fusion_det_ckpt):
        det_model.load_state_dict(torch.load(fusion_det_ckpt, map_location=dev, weights_only=False)['model'])
        det_model.eval()
    
    slda_bin = StreamingLDA(256, 2, device=dev)
    if os.path.exists(slda_binary_ckpt):
        slda_bin.load(slda_binary_ckpt)
        
    # 2. Event Recognition (Optional)
    slda_14 = None
    if fusion_rec_ckpt and slda_14class_ckpt:
        rec_model = PolyGuidedFusion(nc=14).to(dev)
        if os.path.exists(fusion_rec_ckpt):
            rec_model.load_state_dict(torch.load(fusion_rec_ckpt, map_location=dev, weights_only=False)['model'])
            rec_model.eval()
        
        slda_14 = StreamingLDA(256, 14, device=dev)
        if os.path.exists(slda_14class_ckpt):
            slda_14.load(slda_14class_ckpt)
            # Link for inference
            setattr(slda_14, "_rec_model", rec_model)
            
    return det_model, slda_bin, slda_14
