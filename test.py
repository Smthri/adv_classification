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
from concat_dataset import ConcatDatasets


def get_args():
    parser = argparse.ArgumentParser(description='Test a classifier on image folder.')
    parser.add_argument('data_dir', metavar='data_dirs', type=str, nargs='+', help='Path to dataset roots.')
    parser.add_argument('--adv_dir', type=str, default=None, help='Path to adversarial generator. Must accept batches of images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on.')
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

    test_dataset = ConcatDatasets(args.data_dir, transform=get_transforms())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8)

    classes = test_dataset.classes

    device = torch.device(args.device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_dir, map_location=device))
    model.eval()
    
    resize = transforms.Resize(224)

    correct = 0
    total = len(test_dataset)
    
    if args.adv_dir is not None:
        adv_generator = Generator(224, mixed_precision=False)
        adv_generator.load_state_dict(torch.load(args.adv_dir, map_location=torch.device(device)))
        adv_generator.eval()
        with open('gen_summary.txt', 'w') as f:
            f.write(str(summary(adv_generator, input_data=[torch.randn(1, 3, 224, 224), torch.Tensor([1])])))
        adv_generator.to(device)
    else:
        adv_generator = None
    
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
