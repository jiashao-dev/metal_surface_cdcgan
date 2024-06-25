import argparse
import os
import time
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
import torch

from torchmetrics.image.fid import FrechetInceptionDistance as FID

fid = FID(192)

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

print("----------------------------")
print("         Parameter")
print("----------------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.52, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=6, help="number of classes")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")

opt = parser.parse_args()
print(opt)

print("----------------------------\n\n")

def interpolation(imgs):
    arr = []
    for img in imgs:
        img = transforms.ToPILImage()(img)
        img = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])(img)
        arr.append((img * 255).to(dtype=torch.uint8))

    res = torch.stack(arr)
    return res

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 512 * 8 ** 2),
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 3, 2, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, input, label):
        input = torch.concat([input, label], dim=1)
        out = self.l1(input)
        out = out.view(out.shape[0], 512, 8, 8)
        img = self.conv_blocks(out)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(opt.n_classes, opt.img_size * opt.img_size * 1),
            nn.LeakyReLU()
        )

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels + 1, 64, 3, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),

            nn.Conv2d(128, 256, 3, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256, 0.8),

            nn.Conv2d(256, 512, 3, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(512, 0.8),

            nn.Flatten()
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(512 * 8 ** 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, img, label):
        label = self.linear(label).view(label.shape[0], opt.channels, opt.img_size, opt.img_size)
        out = torch.concat([img, label], dim=1)
        out = self.model(out)
        validity = self.adv_layer(out)
        return validity

generator = torch.load("./model/20240606-193850.pth")

dataset = datasets.ImageFolder(
    "./data/metal_surface/",
    transform=transforms.Compose(
        [
            transforms.Resize(opt.img_size),
            transforms.Grayscale(),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ],
    )
)
current_time_str = time.strftime("%Y%m%d-%H%M%S")

## evaluate result
print("----------------------------")
print("         Evaluation")
print("----------------------------\n")

path_to_save_result = "./result/metal_surface/%s/" % (current_time_str)



# generate images
noise = torch.tensor(
    np.random.normal(0, 1, (len(dataset), opt.latent_dim)),
    dtype=torch.float32,
    device=device
)

labels = [i for i in range(6) for _ in range(276)]
labels = F.one_hot(torch.arange(6, device='cuda'), 6)[labels].float()

gen_imgs = generator(noise, labels)

fid_scores = []

for i, class_label in enumerate(["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]):
    path_to_class = path_to_save_result + class_label + "/"

    os.makedirs(path_to_class, exist_ok=True)

    indices = [idx for idx, target in enumerate(dataset.targets) if target == i]

    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=True
    )

    real_imgs, _ = next(iter(dataloader))
    save_image(real_imgs[:9], path_to_class + "real.png", nrow=9, normalize=True)
    
    current_class_gen_imgs = gen_imgs[(i * 276):(i * 276 + 276)]
    save_image((current_class_gen_imgs.data)[:9], path_to_class + "fake.png", nrow=9, normalize=True)

    fid.update(interpolation(real_imgs), real=True)
    fid.update(interpolation(current_class_gen_imgs), real=False)
    result = fid.compute()
    fid.reset()

    fid_scores.append(result)
    print("FID Score (%s): %.4f" % (class_label, result))

print("Overall FID Score: %.4f" % ((sum(fid_scores)/len(fid_scores))))

print("----------------------------\n\n")