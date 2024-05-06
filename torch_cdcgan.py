import argparse
import os
import time
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import torch.nn as nn
import torch

from torchsummary import summary

import torch.nn.functional as F

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

print("----------------------------")
print("         Parameter")
print("----------------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_sample", type=str, default="metal_surface", help="dataset to be used")
parser.add_argument("--n_epochs", type=int, default=1500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=6, help="number of classes")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
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

        def generator_block(in_filters, out_filters, bn=True):
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))

            block.append(nn.ReLU(inplace=True))
            return block

        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 256 * opt.init_size ** 2),
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            *generator_block(256, 128, True),
            *generator_block(128, 64, True),
            *generator_block(64, 32, True),
            *generator_block(32, 16, True),

            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, input, label):
        input = torch.concat([input, label], dim=1)
        out = self.l1(input)
        out = out.view(out.shape[0], 256, opt.init_size, opt.init_size)
        img = self.conv_blocks(out)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(opt.n_classes, opt.img_size * opt.img_size * 1),
            nn.LeakyReLU()
        )

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(opt.channels + 1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Flatten(),
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(256 * opt.init_size ** 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, img, label):
        label = self.linear(label).view(label.shape[0], opt.channels, opt.img_size, opt.img_size)
        out = torch.concat([img, label], dim=1)
        out = self.model(out)
        validity = self.adv_layer(out)
        return validity
    
# get and store generated images
def sample_image(n_row, batches_done, path_to_generate):
    # Sample noise
    z = torch.tensor(
        np.random.normal(0, 1, (n_row * opt.n_classes, opt.latent_dim)),
        dtype=torch.float32,
        device=device
    )
    
    labels = [i // 9 for i in range(54)]
    labels = F.one_hot(torch.arange(6, device="cuda"), 6)[labels].float()

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, path_to_generate + "/%d.png" % batches_done, nrow=n_row, normalize=True)

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
path_to_generate = "./generate/%s/all/%s" % (opt.dataset_sample, current_time_str)

os.makedirs(path_to_generate, exist_ok=True)

start = time.time()
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        generator.train(True)
        discriminator.train(True)

        batch_size = imgs.shape[0]

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
        
        # Configure input
        real_imgs = imgs.clone().detach().to(device=device, dtype=torch.float32)
        labels = labels.clone().detach().to(device=device, dtype=torch.long)

        # ----------- Train Generator -----------

        # Sample noise as generator input
        z = torch.tensor(
            data=np.random.normal(0, 1, (batch_size, opt.latent_dim)),
            dtype=torch.float32,
            device=device
        )

        gen_labels = F.one_hot(torch.arange(opt.n_classes, device=device), opt.n_classes)[labels].float()

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        
        optimizer_generator.zero_grad()

        g_loss.backward()
        optimizer_generator.step()

        # -------- Train Discriminator --------

        ## measure discriminator's ability to classify real from generated samples

        # Loss for real images
        validity_real = discriminator(real_imgs, gen_labels)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)


        outputs = torch.cat((validity_real, validity_fake), 0)
        targets = torch.cat((valid, fake), 0)

        optimizer_discriminator.zero_grad()

        d_loss = adversarial_loss(outputs, targets)
        d_loss.backward()
        optimizer_discriminator.step()

        print(
            "[Epoch %04d/%04d] [Batch %02d/%02d] [D loss: %f] [G loss: %f]"
            % (epoch+1, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if (batches_done % opt.sample_interval == 0):
            sample_image(9, batches_done, path_to_generate)

path_to_save_model = "./model/%s/all" % (opt.dataset_sample)
os.makedirs(path_to_save_model, exist_ok=True)

torch.save(generator, "%s/%s.h5" % (path_to_save_model, current_time_str))
print("----------------------------\n\n")


## evaluate result
print("----------------------------")
print("         Evaluation")
print("----------------------------\n")

path_to_save_result = "./result/%s/%s/" % (current_time_str, opt.dataset_sample)

for i, class_name in enumerate(["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]):
    os.makedirs(path_to_save_result + class_name, exist_ok=True)
    
    indices = [idx for idx, target in enumerate(dataset.targets) if target == i]

    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=9,
        shuffle=True
    )

    imgs, labels = next(iter(dataloader))
    grid = make_grid(imgs, nrow=3)
    save_image(grid, path_to_save_result + "%s/original.png" % (class_name), normalize=True)

    z = torch.tensor(
        np.random.normal(0, 1, (3 ** 2, opt.latent_dim)),
        dtype=torch.float32,
        device=device
    )

    labels = F.one_hot(torch.arange(6, device='cuda'), 6)[labels].float()

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, path_to_save_result + "%s/fake.png" % (class_name), nrow=3, normalize=True)

    for j, gen_img in enumerate(gen_imgs):
        save_image(gen_img.data, path_to_save_result + "%s/%d.png" % (class_name, j+1), normalize=True)

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