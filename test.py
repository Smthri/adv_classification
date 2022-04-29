import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse
from torchvision import transforms
from torchinfo import summary
from torch.nn import functional as F
from pathlib import Path
from skimage.io import imread, imsave
from PIL import Image
import numpy as np
from advgan import *
from torchvision.utils import save_image
from matplotlib import pyplot as plt


class AdvHEDataset(torch.utils.data.Dataset):
    def __init__(self, orig_image_folder, adv_noise_folder=None, factor = 0., transform=None):
        self.orig_image_folder = Path(orig_image_folder)
        self.adv_noise_folder = Path(adv_noise_folder) if adv_noise_folder is not None else None
        self.classes = []#list(range(len([x for x in self.orig_image_folder.iterdir() if x.is_dir()])))
        self.imgs = []
        self.noise = []
        self.transform = transform
        self.factor = factor
        for i, child in enumerate(sorted(self.orig_image_folder.iterdir())):
            self.classes.append(child.name)
            for name in child.iterdir():
                if name.is_file():
                    self.imgs.append((str(name), i))
        self.counter = 0

    def __len__(self):
        return len(self.imgs)

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
    parser.add_argument('--adv_dir', type=str, default=None, help='Path to adv generator')
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

    #test_dataset = AdvHEDataset(args.data_dir, args.adv_dir, args.factor, get_transforms())
    test_dataset = ImageFolder(args.data_dir, transform=get_transforms())
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
    
    resize = transforms.Resize(224)

    #print(summary(model, (3, 224, 224), device='cuda'))

    correct = 0
    total = len(test_dataset)
    
    if args.adv_dir is not None:
        adv_generator = Generator(224, mixed_precision=False)
        adv_generator.load_state_dict(torch.load(args.adv_dir, map_location=torch.device(device)))
        adv_generator.eval()
        adv_generator.to(device)
    else:
        adv_generator = None
        
    with open('gen_summary.txt', 'w') as f:
        f.write(str(summary(adv_generator, input_data=[torch.randn(1, 3, 224, 224), torch.Tensor([1])], device='cuda')))
    
    a = 0
    seen = set()
    class_successful_attack_amount = [0] * len(classes)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(resize(inputs))
            outputs = F.softmax(outputs, 1)
            clean_conf, clean_preds = torch.max(outputs, 1)

            noise = torch.zeros_like(inputs)
            if adv_generator is not None:
                noise = adv_generator(inputs, labels)
                outputs = model(resize(inputs + noise))
                outputs = F.softmax(outputs, 1)
            '''if adv_generator is None:
                outputs = model(resize(inputs))
            else:
                noise = adv_generator(inputs, labels)
                outputs = model(resize(inputs + noise))'''

            conf, preds = torch.max(outputs, 1)
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct += 1
            
            for i, info in enumerate(zip(labels, clean_preds, preds)):
                true_label, clean_pred, adv_pred = info
                if clean_pred == true_label and adv_pred != true_label:
                    class_successful_attack_amount[true_label] += 1
            
            if adv_generator is not None:
                for i, info in enumerate(zip(labels, clean_preds, preds)):
                    true_label, clean_pred, adv_pred = info
                    if (clean_pred == true_label) and (adv_pred != true_label) and not (int(clean_pred) in seen):
                        seen.add(int(clean_pred))

                        to_save = torch.clone(inputs[i])
                        to_save *= torch.Tensor([0.229, 0.224, 0.225])[:, None, None].to(device)
                        to_save += torch.Tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
                        to_save = torch.clamp(to_save, 0., 1.)

                        clean_img = to_save.cpu().numpy()
                        clean_img = np.transpose(clean_img, (1, 2, 0))

                        to_save = torch.clone(noise[i])
                        to_save -= to_save.min()
                        to_save /= (to_save.max() + 0.001)
                        to_save = torch.clamp(to_save, 0., 1.)

                        noise_img = to_save.cpu().numpy()
                        noise_img = np.transpose(noise_img, (1, 2, 0))

                        to_save = torch.clone((inputs + noise)[i])
                        to_save *= torch.Tensor([0.229, 0.224, 0.225])[:, None, None].to(device)
                        to_save += torch.Tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
                        to_save = torch.clamp(to_save, 0., 1.)

                        adv_img = to_save.cpu().numpy()
                        adv_img = np.transpose(adv_img, (1, 2, 0))

                        plt.figure(figsize=(9, 3))
                        plt.subplot(131)
                        plt.imshow(clean_img)
                        plt.xticks([])
                        plt.yticks([])
                        plt.xlabel('Верное предсказание')
                        plt.title(f'Class: {classes[true_label]}, Conf: {clean_conf[i] * 100:.2f}%')

                        plt.subplot(132)
                        plt.imshow(noise_img)
                        plt.xticks([])
                        plt.yticks([])
                        plt.title(f'Noise (normed)')

                        plt.subplot(133)
                        plt.imshow(adv_img)
                        plt.xticks([])
                        plt.yticks([])
                        plt.xlabel('Неверное предсказание')
                        plt.title(f'Class: {classes[adv_pred]}, Conf: {conf[i] * 100:.2f}%')
                        plt.savefig(f'example_{classes[true_label]}.png')
            
            if adv_generator is not None and a == 0:
                to_save = torch.clone(inputs[:16])
                to_save *= torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
                to_save += torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
                to_save = torch.clamp(to_save, 0., 1.)
                save_image(to_save, 'clean.png', nrow=4)
                
                to_save = torch.clone(noise[:16])
                to_save -= to_save.min()
                to_save /= (to_save.max() + 0.001)
                to_save = torch.clamp(to_save, 0., 1.)
                save_image(to_save, 'noise.png', nrow=4)
                
                to_save = torch.clone((inputs + noise)[:16])
                to_save *= torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
                to_save += torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()
                to_save = torch.clamp(to_save, 0., 1.)
                save_image(to_save, 'adversarial.png', nrow=4)
                with open('label_info.txt', 'w') as f:
                    f.write(f'true labels: {str(labels.cpu())}\npredicted_labels" {str(preds.cpu())}')
                a = 1

    with open('class_successful_attack_amount.txt', 'w') as f:
        f.write(str(class_successful_attack_amount))
    plt.figure(figsize=(10, 5))
    plt.bar(classes, class_successful_attack_amount)
    plt.grid(True)
    plt.yticks(class_successful_attack_amount)
    plt.savefig('class_successful_attack_distribution.png')
    acc = correct / total * 100

    print(f'Finished. Accuracy: {acc:.2f}')
