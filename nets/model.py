import torch
from torch import nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.block1 = _conv_block(in_ch, base_ch)
        self.block2 = _conv_block(base_ch, base_ch * 2)
        self.block3 = _conv_block(base_ch * 2, base_ch * 3)
        self.block4 = _conv_block(base_ch * 3, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.block1(x)          # H, W
        x2 = self.block2(self.pool(x1))  # H/2, W/2
        x3 = self.block3(self.pool(x2))  # H/4, W/4
        x4 = self.block4(self.pool(x3))  # H/8, W/8
        return x1, x2, x3, x4


class DetectorBranch(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.block1 = _conv_block(in_ch, base_ch)
        self.block2 = _conv_block(base_ch, base_ch * 2)
        self.block3 = _conv_block(base_ch * 2, base_ch * 3)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch * 3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 3, base_ch * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, 65, kernel_size=1),
        )

    def forward(self, x_gray):
        x = self.block1(x_gray)
        x = self.block2(self.pool(x))
        x = self.block3(self.pool(x))
        x = self.pool(x)  # H/8
        logits = self.head(x)
        return logits


class ReliabilityDecoder(nn.Module):
    def __init__(self, in_ch=128, skip1_ch=32, skip2_ch=64, skip3_ch=96, sem_ch=64):
        super().__init__()
        self.sem_proj = nn.Conv2d(in_ch, sem_ch, kernel_size=1)
        self.up3 = _conv_block(in_ch + skip3_ch, 128)
        self.up2 = _conv_block(128 + skip2_ch, 96)
        self.up1 = _conv_block(96 + skip1_ch, 64)
        self.head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        sem_feat = self.sem_proj(x4)

        u3 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, x3], dim=1))

        u2 = F.interpolate(u3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1))

        u1 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1))

        rel_logits = self.head(u1)
        return rel_logits, sem_feat


class SGFM(nn.Module):
    def __init__(self, desc_ch=256, sem_ch=64, num_heads=4):
        super().__init__()
        self.sem_proj = nn.Conv2d(sem_ch, desc_ch, kernel_size=1)
        self.attn = nn.MultiheadAttention(desc_ch, num_heads=num_heads, batch_first=False)
        self.out_proj = nn.Conv2d(desc_ch, desc_ch, kernel_size=1)

    def forward(self, desc_map, sem_map):
        # desc_map: B, C, H, W
        # sem_map:  B, Cs, H, W
        desc = desc_map
        sem = self.sem_proj(sem_map)
        b, c, h, w = desc.shape
        n = h * w

        q = sem.flatten(2).permute(2, 0, 1)  # N, B, C
        k = desc.flatten(2).permute(2, 0, 1)
        v = k
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(1, 2, 0).reshape(b, c, h, w)

        fused = self.out_proj(desc + attn_out)
        return fused


def sample_descriptors(desc_map, keypoints_xy, image_size):
    # keypoints_xy: B, N, 2 in image coordinates (x, y)
    # image_size: (H, W)
    b, c, h, w = desc_map.shape
    img_h, img_w = image_size

    # Normalize keypoints to [-1, 1] for grid_sample
    x = keypoints_xy[..., 0] / max(img_w - 1, 1) * 2 - 1
    y = keypoints_xy[..., 1] / max(img_h - 1, 1) * 2 - 1
    grid = torch.stack([x, y], dim=-1)
    grid = grid.view(b, -1, 1, 2)

    sampled = F.grid_sample(desc_map, grid, mode="bilinear", align_corners=False)
    sampled = sampled.squeeze(3).permute(0, 2, 1)  # B, N, C
    return F.normalize(sampled, p=2, dim=-1)


class SFD2Chap03Model(nn.Module):
    def __init__(self, base_ch=32, desc_dim=256, sem_ch=64):
        super().__init__()
        self.encoder = Encoder(in_ch=3, base_ch=base_ch)
        self.detector = DetectorBranch(in_ch=1, base_ch=max(base_ch // 2, 8))
        self.rel_decoder = ReliabilityDecoder(
            in_ch=base_ch * 4,
            skip1_ch=base_ch,
            skip2_ch=base_ch * 2,
            skip3_ch=base_ch * 3,
            sem_ch=sem_ch,
        )
        self.desc_proj = nn.Conv2d(base_ch * 4, desc_dim, kernel_size=1)
        self.sgfm = SGFM(desc_ch=desc_dim, sem_ch=sem_ch, num_heads=4)

    def forward(self, x):
        # x: B, 3, H, W
        x1, x2, x3, x4 = self.encoder(x)

        # Repeatability branch (grayscale)
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        rep_logits = self.detector(x_gray)

        # Reliability branch
        rel_logits, sem_feat = self.rel_decoder(x1, x2, x3, x4)

        # Descriptor branch
        desc_map = self.desc_proj(x4)
        sem_feat_up = sem_feat
        if sem_feat.shape[-2:] != desc_map.shape[-2:]:
            sem_feat_up = F.interpolate(sem_feat, size=desc_map.shape[-2:], mode="bilinear", align_corners=False)
        desc_map = self.sgfm(desc_map, sem_feat_up)
        desc_map = F.normalize(desc_map, p=2, dim=1)

        return {
            "rep_logits": rep_logits,
            "rel_logits": rel_logits,
            "desc_map": desc_map,
            "sem_feat": sem_feat,
        }
