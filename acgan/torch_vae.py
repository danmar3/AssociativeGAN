# Torch VAE example used as reference: https://github.com/pytorch/examples/tree/master/vae
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from acgan import torch_gan

import torch.utils.data

import numpy as np
import attr
from tqdm.auto import tqdm

class VAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, z_size=10):
        super(VAE, self).__init__()

        self.input_size = input_size

        #### Encoding Params
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.encoded_means = nn.Linear(hidden_size, z_size)
        self.encoded_var = nn.Linear(hidden_size, z_size)

        #### Decoding Params
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        h_enc = self.encoder(x)
        return self.encoded_means(h_enc), self.encoded_var(h_enc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class CVAE(VAE):
    def __init__(self, input_size, z_size=2, hidden_size=64):
        super(VAE, self).__init__()

        self.input_size = input_size

        enc = torch_gan.make_model_from_block(torch_gan.CNNBlock,
                                               z_dim=3,
                                               n_channels=[hidden_size]*4,
                                               kernel_sizes=[4, 4, 4, 4],
                                               strides=[1, 2, 2],
                                               paddings=[1, 1, 1]
                                               )
        enc.add_module(name='output_conv', module=torch.nn.Conv2d(hidden_size, 1, 4, stride=1))
        enc.add_module(name='output_activ', module=torch.nn.Sigmoid())

        t_rng_arr = torch.rand(1, *self.input_size)  # .reshape(batch_size, *img_shape[1:])
        lin_in_size = np.product(enc(t_rng_arr).shape[1:])
        enc.add_module(name='reshape', module=Reshape((-1, lin_in_size)))

        ####
        #enc.add_module(name='embed_lin', module=torch.nn.Linear(in_features=lin_in_size,
        #                                                        out_features=z_size))
        #enc.add_module(name='embed_act', module=torch.nn.ReLU())

        self.encoder = enc

        #####
        ## Decoder
        dec = torch_gan.make_model_from_block(torch_gan.CNNTransposeBlock,
                                              z_dim=2,
                                              n_channels=[hidden_size]*4,#, 3],
                                              kernel_sizes=[4, 4, 4, 2],#, 2],
                                              strides=[1, 1, 2, 2],
                                              paddings=[0, 0, 0, 0])
        dec.add_module(name='dec_output',
                       module=torch_gan.CNNTransposeBlock(hidden_size, 3, 2, 2,
                                                          activation=torch.nn.Sigmoid()))
        self.decoder = dec


        #### Encoding Params
        #self.encoder = nn.Sequential(
        #    nn.Linear(input_size, hidden_size),
        #    nn.ReLU(),
        #)
        self.encoded_means = nn.Linear(lin_in_size, z_size)
        self.encoded_var = nn.Linear(lin_in_size, z_size)

        #### Decoding Params
        #self.decoder = nn.Sequential(
        #    nn.Linear(z_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, input_size),
        #    nn.Sigmoid()
        #)

    def encode(self, x):
        h_enc = self.encoder(x)
        return self.encoded_means(h_enc), self.encoded_var(h_enc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        #t_rng_embed.reshape(64, 2, 1, 1)
        return self.decoder(z.reshape(-1, 2, 1, 1))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, *self.input_size))
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

    # Reconstruction + KL divergence
    # losses summed over all elements and batch
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        #image_size = 64
        #depth = 3
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, (image_size ** 2) * depth), reduction='sum')
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

