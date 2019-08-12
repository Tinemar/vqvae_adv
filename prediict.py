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


def predict(loader, model,device, args):
    loader = tqdm(loader)

    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        label = label.to(device)
        target_label = target_label.to(device)
        criterion_t = nn.CrossEntropyLoss()

        if args.out == True:
            target_model.eval()
            for i in range(128):
                i = img[i:i+1]
                sample = i
                with torch.no_grad():
                    out, _ = model(sample)
                # utils.save_image(
                #     torch.cat([sample, out], 0),
                #     f'sample_adv/out.png',
                #     nrow=1,
                #     normalize=True,
                #     range=(-1, 1),
                # )
                # final_labels = unpickle('F:/vqvae_adv/cifar-100-batches-py/cifar-100-python/meta')
                # final_labels = final_labels[b'fine_label_names']

                outputs_t = target_model(Variable(out))
                outputs_s = target_model(Variable(sample))
                _, predicted1 = torch.max(outputs_t.data, 1)
                _, predicted2 = torch.max(outputs_s.data, 1)
                print('out:',predicted1,'target_out:',predicted2)
            exit()

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
        loss = recon_loss + latent_loss_weight * latent_loss 
        + adv_loss_weight * adv_loss
        print(loss.mean()) 
        loss.backward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)

    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=bool, default=False)
    parser.add_argument('--ckp', type=str, default='')

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
    if args.ckp != '':
        model = VQVAE()
        model.load_state_dict(torch.load(args.ckp))
        model = model.to(device)
    else:
        model = VQVAE().to(device)
    target_model = models.resnet50(pretrained=True)
    target_model = target_model.to(device)

    predict(loader,target_model,device,args)