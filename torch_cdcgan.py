import argparse
import os
import time
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch

from torchsummary import summary

import torch.nn.functional as F

from torchmetrics.image.fid import FrechetInceptionDistance as FID

fid = FID(192)

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

print("----------------------------")
print("         Parameter")
print("----------------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_sample", type=str, default="metal_surface", help="dataset to be used")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.53, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=6, help="number of classes")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--init_size", type=int, default=8, help="generator initial size")

opt = parser.parse_args()
print(opt)

print("----------------------------\n\n")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 512 * opt.init_size ** 2),
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
        out = out.view(out.shape[0], 512, opt.init_size, opt.init_size)
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
            nn.Linear(512 * opt.init_size ** 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, img, label):
        label = self.linear(label).view(label.shape[0], opt.channels, opt.img_size, opt.img_size)
        out = torch.concat([img, label], dim=1)
        out = self.model(out)
        validity = self.adv_layer(out)
        return validity
    
# get and store generated images
def sample_image(path_to_generate):
    gen_imgs = generator(fixed_noise, fixed_labels)
    save_image(gen_imgs.data, path_to_generate, nrow=9, normalize=True)
    return gen_imgs

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

    return torch.stack(arr)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

print("----------------------------")
print("      Dataset Summary")
print("----------------------------\n")

# Configure data loader
dataset = datasets.ImageFolder(
    "./data/%s/" % (opt.dataset_sample),
    transform=transforms.Compose(
        [
            transforms.Resize(opt.img_size),
            transforms.Grayscale(),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ],
    )
)

dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

print(dataset)
print("----------------------------\n\n")

# Optimizers
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Print model summary
print("----------------------------")
print("     Generator Summary")
print("----------------------------\n")
tst = torch.rand(size=(1,opt.latent_dim), device=device)
lbl = torch.rand(size=(1,opt.n_classes), device=device)
summary(generator, tst, lbl)
print("----------------------------\n\n")


print("----------------------------")
print("    Discriminator Summary")
print("----------------------------\n")
imgs = torch.rand(size=(1, 1, opt.img_size, opt.img_size), device=device)
summary(discriminator, imgs, lbl)
print("----------------------------\n\n")

print("----------------------------")
print("          Training")
print("----------------------------\n")

current_time_str = time.strftime("%Y%m%d-%H%M%S")
path_to_generate = "./generate/%s/%s" % (opt.dataset_sample, current_time_str)

os.makedirs(path_to_generate, exist_ok=True)

# Fixed sample noise
fixed_noise = torch.tensor(
    np.random.normal(0, 1, (9 * opt.n_classes, opt.latent_dim)),
    dtype=torch.float32,
    device=device
)

fixed_labels = [i // 9 for i in range(54)]
fixed_labels = F.one_hot(torch.arange(6, device="cuda"), 6)[fixed_labels].float()

start = time.time()
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        generator.train()
        discriminator.train()

        # Configure input
        batch_size = imgs.shape[0]
        real_imgs = imgs.clone().detach().to(device=device, dtype=torch.float32)
        gen_labels = labels.clone().detach().to(device=device, dtype=torch.long)
        gen_labels = F.one_hot(torch.arange(opt.n_classes, device=device), opt.n_classes)[gen_labels].float()

        # Adversarial ground truths
        # apply label smoothing
        valid = torch.tensor(
            data=[[0.9]] * batch_size,
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )

        fake = torch.tensor(
            data=[[0.1]] * batch_size,
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )

        # -------- Train Discriminator --------
        optimizer_discriminator.zero_grad()

        ## measure discriminator's ability to classify real from generated samples

        # Loss for real images
        validity_real = discriminator(real_imgs, gen_labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        d_real_loss.backward()

        # Loss for fake images
        z = torch.tensor(
            data=np.random.normal(0, 1, (batch_size, opt.latent_dim)),
            dtype=torch.float32,
            device=device
        )

        gen_imgs = generator(z, gen_labels)

        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss

        optimizer_discriminator.step()


        # ----------- Train Generator -----------
        optimizer_generator.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_generator.step()


        summary_statistics = "[Epoch %04d/%04d] [Batch %02d/%02d] [D loss: %.4f] [G loss: %.4f]" % (epoch+1, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item())

        batches_done = epoch * len(dataloader) + i
        if ((epoch+1 == opt.n_epochs and i+1 == len(dataloader)) or (batches_done % opt.sample_interval == 0)):
            sample_image(path_to_generate + "/%d.png" % batches_done)
            fid.update(interpolation(real_imgs), real=True)
            fid.update(interpolation(gen_imgs), real=False)
            fid_score = fid.compute()
            summary_statistics += " [FID: %.4f]" % fid_score
            fid.reset()

        print(summary_statistics)

path_to_save_model = "./model/%s/" % (opt.dataset_sample)
os.makedirs(path_to_save_model, exist_ok=True)

torch.save(generator, "%s/%s.h5" % (path_to_save_model, current_time_str))
print("----------------------------\n\n")


## evaluate result
print("----------------------------")
print("         Evaluation")
print("----------------------------\n")

path_to_save_result = "./result/%s/%s/" % (opt.dataset_sample, current_time_str)

os.makedirs(path_to_save_result, exist_ok=True)

batch_size = 9 * opt.n_classes

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

noise = torch.tensor(
    np.random.normal(0, 1, (batch_size, opt.latent_dim)),
    dtype=torch.float32,
    device=device
)

real_imgs, real_labels = next(iter(dataloader))
save_image(real_imgs, path_to_save_result + "real.png", nrow=9, normalize=True)

real_labels = F.one_hot(torch.arange(6, device='cuda'), 6)[real_labels].float()
gen_imgs = generator(noise, real_labels)

save_image(gen_imgs.data, path_to_save_result + "fake.png", nrow=9, normalize=True)

fid.update(interpolation(real_imgs), real=True)
fid.update(interpolation(gen_imgs), real=False)
result = fid.compute()
fid.reset()

print("FID Score: %.4f" % (result))

print("----------------------------\n\n")

end = time.time()
diff = end - start

second = 0
minute = diff // 60
hour = 0

if minute > 60:
    hour = minute // 60
    minute %= 60

second = diff % 60

print("\nTime used: %d hours %d minutes %d seconds\n" % (hour, minute, second))