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

from lookahead import Lookahead
from vqvae import VQVAE
from scheduler import CycleScheduler


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


def train(epoch, loader,validation_loader, model, target_model, target_label, optimizer, scheduler, device, args):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    adv_loss_weight = 0.25
    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    if args.out == True:
        for i, (img, label) in enumerate(validation_loader):
            model.zero_grad()
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            print(label)
            target_label = target_label.to(device)
            criterion_t = nn.CrossEntropyLoss().to(device)
            target_model.eval()
            suc = 0
            for i in range(128):
                i = img[i:i+1]
                label = label[0]
                sample = i
                with torch.no_grad():
                    out, _ = model(sample)

                outputs_t = target_model(Variable(out))
                outputs_s = target_model(Variable(sample))
                _, predicted1 = torch.max(outputs_t.data, 1)
                _, predicted2 = torch.max(outputs_s.data, 1)
                if predicted1!=predicted2:
                    suc +=1
                # print('out:', predicted1, 'target_out:', predicted2)
            print(suc/128)
        exit()
    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        target_label = target_label.to(device)
        criterion_t = nn.CrossEntropyLoss().to(device)

        # out:vqvae输出的图片信息
        # latent_loss

        # adv_loss 目标类别和vqvae out的损失
        out, latent_loss = model(img)
        with torch.no_grad():
            target_out = target_model(out)
        try:
            adv_loss = criterion_t(target_out, target_label.long())
        except:
            target_label = torch.ones(80)
            target_label = target_label.to(device)
            adv_loss = criterion_t(target_out, target_label.long())
        # 重建损失MSELoss()
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        print(adv_loss)
        loss = recon_loss + latent_loss_weight * latent_loss
        + adv_loss_weight * adv_loss
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
    parser.add_argument('--ckp', type=str, default='')
    parser.add_argument('--tmodel', type=str, default='')

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
    train_dataset = datasets.CIFAR10(
        root=args.path, train=True, download=True, transform=transform)
    validation_data = datasets.CIFAR10(root=args.path, train=False, download=True,
                                        transform=transform)
    loader = DataLoader(train_dataset, batch_size=128,
                        shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=128,
                        shuffle=True, num_workers=4, pin_memory=True)
    if args.ckp != '':
        model = VQVAE()
        model.load_state_dict(torch.load(args.ckp))
        model = model.to(device)
    else:
        model = VQVAE().to(device)

    target_label = torch.ones(128)
    print(target_label.size())
    if args.tmodel != '':
        target_model = torch.load(args.tmodel).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    # optimizer.zero_grad()

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )
    if args.out == True:
        train(1, loader,validation_loader, model, target_model, target_label,
              optimizer, scheduler, device, args)

    torch.cuda.empty_cache()

    for i in range(args.epoch):
        train(i, loader, validation_loader,model, target_model, target_label,
              optimizer, scheduler, device, args)
        torch.save(
            model.state_dict(
            ), f'checkpoint_adv/vqvae_{str(i + 1).zfill(3)}.pt'
        )
