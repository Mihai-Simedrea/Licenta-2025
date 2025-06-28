from torch import nn
from typing import List
from models import BreastImageModel
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class DenseNetBackbone(torch.nn.Module):
    def __init__(self, features, out_channels=1024):
        super().__init__()
        self.body = features
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        return {"0": x}

class BreastCancerDetectionDataset(Dataset):
    def __init__(self, data: List[BreastImageModel], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item.content

        image = torch.tensor(image, dtype=torch.float32) / 255.0
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)

        target = {}

        if item.bbox:
            x, y, w, h = item.bbox
            boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
            labels = torch.tensor([item.label], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return image, target
