import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils

import attr
from tqdm.auto import tqdm


from acgan.torch_gan import auto_extend, make_model_from_block
from acgan.torch_gan import weights_init, intermediate_outputs
from acgan.torch_gan import CNNBlock, CNNTransposeBlock

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils

import attr
from tqdm.auto import tqdm
from torch.nn import functional as F

from acgan.torch_gan import auto_extend, make_model_from_block
from acgan.torch_gan import weights_init, intermediate_outputs
from acgan.torch_gan import CNNBlock, CNNTransposeBlock


class VAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, z_size=10,
                 conditioning_features=0):
        super(VAE, self).__init__()

        self.input_size = input_size

        #### Encoding Params
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.encoded_means = nn.Linear(hidden_size, z_size)
        self.encoded_var = nn.Linear(hidden_size, z_size)
        self.conditioning_features = conditioning_features

        #### Decoding Params
        self.decoder = nn.Sequential(
            nn.Linear(z_size + self.conditioning_features,
                      hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        if self.conditioning_features > 0:
            _x = x[:, :-self.conditioning_features]
            c_x = x[:, -self.conditioning_features:]
        else:
            _x = x
            c_x = None

        h_enc = self.encoder(_x)
        return self.encoded_means(h_enc), self.encoded_var(h_enc), c_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar, c_x = self.encode(x.view(-1, self.input_size + self.conditioning_features))
        z = self.reparameterize(mu, logvar)
        c_zx = torch.cat((z, c_x), 1)
        # return self.decode(z), mu, logvar
        return self.decode(c_zx), mu, logvar


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

    # Reconstruction + KL divergence
    # losses summed over all elements and batch
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        # image_size = 64
        # depth = 3
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, (image_size ** 2) * depth), reduction='sum')
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, (x.shape[1] * x.shape[2] * x.shape[3])),
                                     reduction='sum')

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
                    for batch_idx, (data, c_x) in enumerate(self.data_gen):
                        data = data.to(self.device)
                        self.optim.zero_grad()

                        recon_batch, mu, logvar = self.model(c_x)
                        loss = VAETrainer.loss_function(recon_batch, data, mu, logvar)

                        loss.backward()
                        train_loss += loss.item()
                        self.optim.step()

                        # Save Losses for plotting later
                        losses.append(loss.item())
                        # D_losses.append(errD.item())
                        batch_pbar.set_description("Loss: %.3f"
                                                   % (np.mean(losses[-20:])))
                        batch_pbar.update(1)

                epoch_pbar.update(1)
        return losses

    # def train_l2_cond_loss(self, n_epochs, log_interval)

