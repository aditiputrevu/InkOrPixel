import os
from PIL import Image
from torch.utils.data import Dataset

class ArtworkDataset(Dataset):
    def __init__(self, image_paths=None, labels=None, transform=None):
        self.image_paths = image_paths if image_paths is not None else []
        self.labels = labels if labels is not None else []
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label