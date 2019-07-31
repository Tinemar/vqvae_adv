# -*- coding: UTF-8 -*-

import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from unpickle import unpickle
from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


def train(epoch, loader, model, target_model, target_label, optimizer, scheduler, device, args):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    adv_loss_weight = 0.25
    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        label = label.to(device)
        target_label = target_label.to(device)
        criterion_t = nn.CrossEntropyLoss()

        if args.out == True:
            model.eval()
            sample = img[:1].to(device)
            with torch.no_grad():
                out, _ = model(sample)
            utils.save_image(
                torch.cat([sample, out], 0),
                f'sample_adv/out.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            outputs = target_model(Variable(out))
            _, predicted = torch.max(outputs.data, 1)
            for i in predicted:
                print(i)
            exit()

        # out:vqvae输出的图片信息
        # latent_loss

        # adv_loss 目标类别和vqvae out的损失
        out, latent_loss = model(img)
        with torch.no_grad():
            target_out = target_model(out)
        adv_loss = criterion_t(target_out, target_label.long())
        # 重建损失MSELoss()
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * \
            latent_loss + adv_loss_weight * adv_loss
        print(loss)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f'sample_adv/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)

    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=bool, default=False)
    parser.add_argument('--checkPoint', type=str, default='')
    parser.add_argument('--target', type=str, default='')

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    train_dataset = datasets.CIFAR100(
        root=args.path, train=True, download=True, transform=transform)
    # validation_data = datasets.CIFAR100(root=args.path, train=False, download=True,
    #                                     transform=transform)
    loader = DataLoader(train_dataset, batch_size=128,
                        shuffle=True, num_workers=4, pin_memory=True)
    if args.checkPoint != '':
        model = VQVAE()
        model.load_state_dict(torch.load(args.checkPoint))
        model = model.to(device)
    else:
        model = VQVAE().to(device)

    target_label = torch.Tensor([40,  1, 91, 79, 31,  9, 29, 62, 60, 68,  6, 40, 81, 55, 98, 94, 24, 59,
                                 9,  2,  7, 47, 45,  8, 18,  6, 47, 46, 77, 56, 93, 22, 70, 94, 74, 72,
                                 16, 31, 39, 41, 37, 30, 91, 57, 79, 66, 73, 68, 39, 93, 95, 70, 28, 28,
                                 19, 17, 45, 28, 21, 13, 67, 91,  0, 67, 17, 51, 94, 87,  9, 72, 69, 74,
                                 22, 17, 26, 79, 89, 31, 19, 49, 19, 60, 27, 96, 60, 29, 70, 73, 24, 30,
                                 63, 72, 24, 75, 18, 25, 15, 45, 24, 60, 45, 69, 58, 19, 34, 58, 23, 30,
                                 63, 95,  1, 16, 73, 35, 19, 85, 95, 74, 67, 99, 52, 72, 82, 11, 88, 63,
                                 77, 60])
    target_model = models.resnet50(pretrained=True)
    target_model = target_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )
    if args.out == True:
        train(1, loader, model, target_model, target_label,
              optimizer, scheduler, device, args)

    torch.cuda.empty_cache()

    for i in range(args.epoch):
        train(i, loader, model, target_model, target_label,
              optimizer, scheduler, device, args)
        torch.save(
            model.module.state_dict(
            ), f'checkpoint_adv/vqvae_{str(i + 1).zfill(3)}.pt'
        )
