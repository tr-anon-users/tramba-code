import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, pred_len, L, D)
        return self.fc2(F.relu(self.fc1(x)))  # (B, pred_len, L, 1)


class LearnableSimilarity(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x, y):
        similarity = torch.sum(x * y, dim=-1).unsqueeze(-1)
        return self.linear(similarity)


class ShiftAwareAttention_AllLinks(nn.Module):
    def __init__(self, d_model, max_shift=20, extra_shifts=None, daily_cycle=288, horizon_offset=1):
        super().__init__()
        self.d_model = d_model
        self.max_shift = max_shift
        self.learnable_similarity = LearnableSimilarity(d_model)

        if extra_shifts is None:
            extra_shifts = []
        default_daily_shift = -(daily_cycle - horizon_offset)
        extra_shifts.append(default_daily_shift)

        self.shifts = sorted(set(list(range(-self.max_shift, 1)) + extra_shifts))

    def forward(self, x):
        B, T, L, D = x.shape
        valid_shifts = [s for s in self.shifts if 0 < abs(s) < T]
        out_all = []

        for b in range(B):
            x_b = x[b:b+1]
            x_shifted = []
            for s in valid_shifts:
                pad_len = abs(s)
                pad = torch.zeros(1, pad_len, L, D, device=x.device)
                if s < 0:
                    shifted = torch.cat([x_b[:, pad_len:], pad], dim=1)
                else:
                    shifted = torch.cat([pad, x_b[:, :-pad_len]], dim=1)
                x_shifted.append(shifted)

            if not x_shifted:
                out_all.append(x_b)
                continue

            x_shifted = torch.stack(x_shifted, dim=1)
            S = x_shifted.shape[1]

            target = x_b
            target_exp = target.unsqueeze(1).unsqueeze(4)
            x_shifted_exp = x_shifted.unsqueeze(3)

            sim = self.learnable_similarity(target_exp, x_shifted_exp)
            sim_avg = sim.mean(dim=2)

            attn_weights = F.softmax(sim_avg.view(1, S * L, L), dim=1).view(1, L, S * L)
            x_shifted_flat = x_shifted.permute(0, 2, 1, 3, 4).reshape(1, T, S * L, D)

            output = []
            for l in range(L):
                weight = attn_weights[0, l].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                fused = torch.sum(x_shifted_flat * weight, dim=2)
                output.append(fused.unsqueeze(2))

            out_all.append(torch.cat(output, dim=2))

        return torch.cat(out_all, dim=0)


class Tramba(nn.Module):
    def __init__(self, d_model=32, in_features=3, pred_len=1, nhead=1, num_links=366):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        self.embedding = nn.Linear(in_features, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model)

        self.agg_layer = nn.Linear(d_model, d_model)
        self.adaptive_emb = nn.Linear(d_model, d_model)

        self.spatial_attn = ShiftAwareAttention_AllLinks(d_model, max_shift=20)
        self.norm1 = nn.LayerNorm(d_model)

        self.temporal_mamba = Mamba(MambaConfig(d_model=d_model, n_layers=2))
        self.ffn = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.fusion_gate = nn.Parameter(torch.zeros(num_links, 1))
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, L, F)
        B, T, L, F = x.shape

        x = self.embedding(x)
        pos = torch.arange(T, device=x.device)
        pos_enc = self.pos_embedding(pos).unsqueeze(0).unsqueeze(2)
        x = x + pos_enc
        x = self.agg_layer(x)

        x_attn = self.spatial_attn(x)
        x_attn = self.norm1(x_attn)
        x_attn_last = x_attn[:, -1, :, :]

        x_mamba = x.permute(0, 2, 1, 3).reshape(B * L, T, -1)
        x_mamba_out = self.temporal_mamba(x_mamba)
        x_mamba_out = self.ffn(x_mamba_out) + x_mamba_out
        x_mamba_out = self.norm2(x_mamba_out)
        x_mamba_last = x_mamba_out[:, -1, :].view(B, L, -1)

        gate = torch.sigmoid(self.fusion_gate).unsqueeze(0)
        x_fused = gate * x_attn_last + (1 - gate) * x_mamba_last

        x_out = self.fc(x_fused).unsqueeze(1)
        return x_out.repeat(1, self.pred_len, 1, 1)
