import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from acgan import torch_gan

import torch.utils.data
import torchvision.datasets as dset

import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt

import numpy as np
import attr
from tqdm.auto import tqdm



class VAE(nn.Module):
    def __init__(self, input_size=784):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


@attr.attrs
class VAETrainer:
    model = attr.ib()

    data_gen = attr.ib()
    learning_rate = attr.ib(0.0002)
    beta1 = attr.ib(0.5)
    n_samples = attr.ib(None)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    criterion = attr.ib(torch.nn.BCELoss())

    epochs_trained = attr.ib(0, init=False)

    def init_weights(self):
        pass

    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        image_size = 64
        depth = 3
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, (image_size ** 2) * depth), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train(self, n_epochs, log_interval=10):

        self.model.to(self.device)
        self.model.train()
        train_loss = 0
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.learning_rate,
                                      betas=(self.beta1, 0.999))

        losses = list()
        with tqdm(total=n_epochs,
                  desc='Training epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):

                with tqdm(total=self.n_samples, desc='-loss-') as batch_pbar:
                    for batch_idx, (data, _) in enumerate(self.data_gen):
                        data = data.to(self.device)
                        self.optim.zero_grad()

                        recon_batch, mu, logvar = self.model(data)
                        loss = VAETrainer.loss_function(recon_batch, data, mu, logvar)

                        loss.backward()
                        train_loss += loss.item()
                        self.optim.step()

                        # Save Losses for plotting later
                        losses.append(loss.item())
                        #D_losses.append(errD.item())
                        batch_pbar.set_description("Loss: %.3f"
                                                   % (np.mean(losses[-20:])))
                        batch_pbar.update(1)

                epoch_pbar.update(1)
        return losses

                #if batch_idx % log_interval == 0:
                #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #        epoch, batch_idx * len(data), len(self.data_gen.dataset),
                #               100. * batch_idx / len(train_loader),
                #               loss.item() / len(data)))

            #print('====> Epoch: {} Average loss: {:.4f}'.format(
            #    epoch, train_loss / len(train_loader.dataset)))


if __name__ == """__main__""":
    image_size = 64
    input_size = (image_size ** 2)

    model = VAE(input_size=input_size)#.to(device)
    VAETrainer()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)

