import torch
import torchvision
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse
from torchvision import transforms
import numpy as np
from datetime import datetime
import os
import wandb


def get_args():
    parser = argparse.ArgumentParser(description='Train a classifier on image folder.')
    parser.add_argument('--data_dir', required=True, type=str, help='Path to dataset root.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on.')
    parser.add_argument('--n_epochs', type=int, default=90, help='Number of epochs.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to save checkpoints.')

    return parser.parse_args()


def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomAffine(degrees=15, translate=(.1, .1), scale=(.8, 1.2))
    ])


if __name__ == '__main__':
    args = get_args()

    dataset = ImageFolder(args.data_dir, transform=get_transforms())
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [16000, 2000], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=32)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=32)

    wandb.init(project='HE_classifier')

    classes = dataset.classes
    print(f'Loaded train dataset with {len(train_dataset)} images.')
    print(f'Classes: {classes}')

    device = torch.device(args.device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    min_loss = np.inf

    for epoch in range(args.n_epochs):
        print(f'Starting epoch {epoch}:')
        losses = []
        accs = []
        val_losses = []
        val_accs = []

        # Train
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # gather statistics
            losses.append(loss.item())

            _, preds = torch.max(outputs, 1)
            correct = 0
            total = inputs.size(0)
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct += 1

            accs.append(correct / total * 100)

        # Validate
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # gather statistics
                val_losses.append(loss.item())

                _, preds = torch.max(outputs, 1)
                correct = 0
                total = inputs.size(0)
                for label, prediction in zip(labels, preds):
                    if label == prediction:
                        correct += 1

                val_accs.append(correct / total * 100)

        mean_loss = np.mean(losses)
        mean_acc = np.mean(accs)
        mean_val_loss = np.mean(val_losses)
        mean_val_acc = np.mean(val_accs)

        print(f'Finished epoch, loss: {mean_loss}, acc: {mean_acc}, val_loss: {mean_val_loss}, val_acc: {mean_val_acc}')

        if mean_loss < min_loss:
            print(f'Loss decreased from {min_loss} to {mean_loss}. Saving model to {args.checkpoint_dir}.')
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pth'))
            min_loss = mean_loss

        wandb.log({
            'train/loss': mean_loss,
            'train/accuracy': mean_acc,
            'val/loss': mean_val_loss,
            'val/accuracy': mean_val_acc
        })
        scheduler.step(mean_loss)
