import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse
from torchvision import transforms
from torchsummary import summary
from pathlib import Path
from skimage.io import imread, imsave
from PIL import Image
import numpy as np


class AdvHEDataset(torch.utils.data.Dataset):
    def __init__(self, orig_image_folder, adv_noise_folder=None, mode='clean', factor = 0., transform=None):
        assert mode in ['adv', 'clean']
        self.orig_image_folder = Path(orig_image_folder)
        self.adv_noise_folder = Path(adv_noise_folder) if adv_noise_folder is not None else None
        self.classes = []#list(range(len([x for x in self.orig_image_folder.iterdir() if x.is_dir()])))
        self.imgs = []
        self.noise = []
        self.mode = mode
        self.transform = transform
        self.factor = factor
        for i, child in enumerate(sorted(self.orig_image_folder.iterdir())):
            self.classes.append(child.name)
            for name in child.iterdir():
                if name.is_file():
                    self.imgs.append((str(name), i))

        if adv_noise_folder is not None:
            for i, child in enumerate(sorted(self.adv_noise_folder.iterdir())):
                for name in child.iterdir():
                    if name.is_file():
                        self.noise.append((str(name), i))

            assert len(self.imgs) == len(self.noise), 'Noise and img datasets nust match length'
        self.counter = 0

    def __len__(self):
        return len(self.imgs)

    def set_mode(self, mode):
        assert mode in ['adv', 'clean']
        self.mode = mode

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        with open(img_name, 'rb') as f:
            img = Image.open(f).convert("RGB")
        #img = imread(img_name)
        if self.transform:
            img = self.transform(img)

        if self.mode == 'adv':
            noise_name, _ = self.noise[idx]
            noise = Image.open(noise_name)
            #noise = imread(noise_name)
            if self.transform:
                noise = self.transform(noise)
            if self.counter < 0:
                print(img.shape)
                self.counter += 1
                to_save = img.numpy()
                to_save = np.transpose(to_save - to_save.min(), (1, 2, 0))
                imsave(f'{label}_{self.counter}.png', to_save / to_save.max() * 255)
                to_save = (img + self.factor * noise).numpy()
                to_save = np.transpose(to_save, (1, 2, 0))
                to_save -= to_save.min()
                imsave(f'adv_{label}_{self.counter}.png', to_save / to_save.max() * 255)

            img = img + self.factor * noise

        return img, label


def get_args():
    parser = argparse.ArgumentParser(description='Test a classifier on image folder.')
    parser.add_argument('--data_dir', required=True, type=str, help='Path to dataset root.')
    parser.add_argument('--noise_dir', type=str, default=None, help='Path to noise dataset root')
    parser.add_argument('--factor', type=float, default=0, help='Factor of adversarial noise')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/best.pth', help='Path to load checkpoint.')

    return parser.parse_args()


def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    args = get_args()

    test_dataset = AdvHEDataset(args.data_dir, args.noise_dir, 'adv', args.factor, get_transforms())
    #test_dataset = ImageFolder(args.data_dir, transform=get_transforms())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8)

    classes = test_dataset.classes
    print(f'Loaded dataset with {len(test_dataset)} images.')
    print(f'Classes: {classes}')

    device = torch.device(args.device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_dir, map_location=device))
    model.eval()

    #print(summary(model, (3, 224, 224), device='cuda'))

    correct = 0
    total = len(test_dataset)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct += 1

    acc = correct / total * 100

    print(f'Finished. Accuracy: {acc:.2f}')
