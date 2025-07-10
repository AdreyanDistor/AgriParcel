
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FarmlandDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.classes = class_names
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for root, _, files in os.walk(cls_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                        samples.append((os.path.join(root, file), cls_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return self[np.random.randint(0, len(self))]
