from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image


class ConcatDatasets(torch.utils.data.Dataset):
    def __init__(self, image_folders, transform=None):
        self.folders = [Path(folder) for folder in image_folders]
        self.classes = [child.name for child in sorted(self.folders[0].iterdir())]
        self.data = []
        self.transform = transform
        for folder in tqdm(self.folders, desc='Reading folders...'):
            for i, child in enumerate(sorted(folder.iterdir())):
                assert child.name in self.classes, 'Currently, datasets with different classes are not supported!'
                for name in child.iterdir():
                    if name.is_file():
                        self.data.append((str(name), i))
        print(f'Successfully loaded datasets with total {len(self.data)} images.\nClasses: {self.classes}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        with open(img_name, 'rb') as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label
    