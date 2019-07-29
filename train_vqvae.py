import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler


def train(epoch, loader, model, optimizer, scheduler, device,args):
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
        criterion_t = nn.CrossEntropyLoss()
        target = args.out
        target_model = models.resnet152(pretrained=True)
        target_model = nn.DataParallel(target_model).to(device)
        if args.out == True:
            model.eval()
            sample = img[:1].to(device)
            with torch.no_grad():
                out,_ = model(sample)
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
            _,predicted = torch.max(outputs.data,1)
            for i in predicted:
                print(i)
            exit()
        with torch.no_grad():
            target_out = target_model(img)
        adv_loss = criterion_t(target_out,label)
        #out:vqvae输出的图片信息
        #latent_loss 
        out, latent_loss = model(img)
        
        #重建损失MSELoss()
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss + adv_loss_weight * adv_loss
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
    parser.add_argument('--out',type=bool,default=False)
    parser.add_argument('--checkPoint',type=str,default='')
    parser.add_argument('--target',type=str,default='')

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
    train_dataset = datasets.CIFAR100(root='F:/vq-vae-2-pytorch/cifar-10-batches-py',train=True,download=True, transform=transform)
    validation_data = datasets.CIFAR100(root="F:/vq-vae-2-pytorch/cifar-10-batches-py", train=False, download=True,
                                  transform=transform)
    loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4)
    if args.checkPoint != '':
        model = VQVAE()
        model.load_state_dict(torch.load(args.checkPoint))
        model = nn.DataParallel(model).to(device)
    else:
        model = nn.DataParallel(VQVAE()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )
    if args.out == True:
        train(1, loader, model, optimizer, scheduler, device, args)
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, args)
        torch.save(
            model.module.state_dict(), f'checkpoint_adv/vqvae_{str(i + 1).zfill(3)}.pt'
        )
