import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
from torchvision import transforms
import numpy as np
from datetime import datetime
import os
from skimage.io import imread, imsave
from pathlib import Path
from advgan import *


def get_args():
    parser = argparse.ArgumentParser(description='Train a classifier on image folder.')
    parser.add_argument('--data_dir', required=True, type=str, help='Path to dataset root.')
    parser.add_argument('--adv_dir', type=str, default=None, help='Path to adv generator')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to save checkpoints.')
    parser.add_argument('--epsilon', type=float, default=0.07, help='Epsilon for FGSM.')
    parser.add_argument('--dst_dir', required=True, type=str, help='Path to save generated images.')

    return parser.parse_args()


def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -2, 2)
    return perturbed_image


class AdvHEDataset(torch.utils.data.Dataset):
    def __init__(self, orig_image_folder, adv_noise_folder=None, mode='clean', transform=None):
        assert mode in ['adv', 'clean']
        self.orig_image_folder = Path(orig_image_folder)
        self.adv_noise_folder = Path(adv_noise_folder) if adv_noise_folder is not None else None
        self.classes = []#list(range(len([x for x in self.orig_image_folder.iterdir() if x.is_dir()])))
        self.imgs = []
        self.noise = []
        self.mode = mode
        self.transform = transform
        for i, child in enumerate(self.orig_image_folder.iterdir()):
            self.classes.append(child.name)
            for name in child.iterdir():
                if name.is_file():
                    self.imgs.append((str(name), i))

        if adv_noise_folder is not None:
            for i, child in enumerate(self.adv_noise_folder.iterdir()):
                for name in child.iterdir():
                    if name.is_file():
                        self.noise.append((str(name), i))

            assert len(self.imgs) == len(self.noise), 'Noise and img datasets nust match length'

    def __len__(self):
        return len(self.imgs)

    def set_mode(self, mode):
        assert mode in ['adv', 'clean']
        self.mode = mode

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        img = imread(img_name)
        if self.transform:
            img = self.transform(img)

        if self.mode == 'adv':
            noise_name, _ = self.noise[idx]
            noise = imread(noise_name)
            if self.transform:
                noise = self.transform(noise)
            img = img + noise

        return img, label


if __name__ == '__main__':
    args = get_args()

    #train_dataset = AdvHEDataset(args.data_dir, args.noise_dir, 'adv', get_transforms())
    train_dataset = ImageFolder(args.data_dir, transform=get_transforms())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=8)

    classes = train_dataset.classes
    print(f'Loaded dataset with {len(train_dataset)} images.')
    print(f'Classes: {classes}')

    device = torch.device(args.device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_dir, map_location=device))
    model.eval()
    
    adv_generator = Generator(224, mixed_precision=False)
    adv_generator.load_state_dict(torch.load(args.adv_dir, map_location=torch.device(device)))
    adv_generator.eval()
    adv_generator.to(device)
      
    counter = 0
    means = np.array([0.485, 0.456, 0.406])[:, None, None]
    stds = np.array([0.229, 0.224, 0.225])[:, None, None]

    # Loop over all examples in test set
    with torch.no_grad():
        for data, target in tqdm(train_loader, total=len(train_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            noise = adv_generator(data, init_pred)
            output = model(data + noise)
            final_pred = output.max(1, keepdim=True)[1]
            
            adv_ex = (data + noise).squeeze().detach().cpu().numpy() * stds + means
            hwc = (np.transpose(adv_ex, (1, 2, 0)) * 255).astype(np.uint8)
            imsave(os.path.join(args.dst_dir, classes[target.item()], 'adv_' + str(counter) + '.png'), hwc)
            counter += 1
