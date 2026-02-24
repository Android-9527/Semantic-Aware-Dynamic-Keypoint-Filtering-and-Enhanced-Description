from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CityscapesBinaryReliabilityDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        rep_label_dir=None,
        resize=None,
        dynamic_ids=None,
        rep_label_ext=".png",
        rep_label_downscale=8,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.rep_label_dir = Path(rep_label_dir) if rep_label_dir else None
        self.resize = resize
        self.dynamic_ids = set(dynamic_ids or [])
        self.rep_label_ext = rep_label_ext
        self.rep_label_downscale = rep_label_downscale

        exts = {".png", ".jpg", ".jpeg"}
        self.image_paths = [p for p in self.image_dir.iterdir() if p.suffix.lower() in exts]
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.resize:
            img = cv2.resize(img, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def _load_label(self, path, image_shape):
        label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(path)
        if label.ndim == 3:
            label = label[:, :, 0]
        if self.resize:
            label = cv2.resize(label, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_NEAREST)

        # Map to binary: 1 for dynamic, 0 for static
        rel = np.zeros_like(label, dtype=np.uint8)
        if self.dynamic_ids:
            for idx in self.dynamic_ids:
                rel[label == idx] = 1
        else:
            rel[label > 0] = 1

        rel = torch.from_numpy(rel.astype(np.int64))
        return rel

    def _load_rep_label(self, path, image_shape):
        if path is None or not path.exists():
            return None
        rep = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rep is None:
            return None
        if rep.ndim == 3:
            rep = rep[:, :, 0]
        if self.resize:
            rep = cv2.resize(rep, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_NEAREST)
        if self.rep_label_downscale and self.rep_label_downscale > 1:
            h, w = rep.shape[:2]
            rep = cv2.resize(rep, (w // self.rep_label_downscale, h // self.rep_label_downscale), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(rep.astype(np.int64))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / img_path.name

        image = self._load_image(img_path)
        rel_label = self._load_label(label_path, image.shape[-2:])

        rep_label = None
        if self.rep_label_dir:
            rep_path = (self.rep_label_dir / img_path.stem).with_suffix(self.rep_label_ext)
            rep_label = self._load_rep_label(rep_path, image.shape[-2:])

        return {
            "image": image,
            "rel_label": rel_label,
            "rep_label": rep_label,
            "name": img_path.name,
        }
