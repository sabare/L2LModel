import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os
from torchvision.utils import save_image
import glob
import math 
import gc
import time
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,img_dim),
            nn.Tanh()
        )
    def forward(self,x):
        return self.gen(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 1

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()


sample_dir = '/content/drive/MyDrive/GAN Images/MNIST'
os.makedirs(sample_dir, exist_ok=True)
#fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

def save_samples(index1, index2, latent_tensors, show=True):
    #fake_images = (latent_tensors,784)
    fake_images = torch.reshape(latent_tensors,(-1,1,28,28))
    fake_fname = 'generated-images-{}-{}.png'.format(index1, index2)
    save_image((fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
            """with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 28, 28)
                plt.imshow(fake[0])"""
                
          
        
        save_samples(epoch+1, (batch_idx+1), fake, show=False)
        


