import numpy as np
import torch
from acgan import torch_gan, evaluation
from acgan import torch_vae
from torch import nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils
from acgan.torch_gan import weights_init

import attr
from tqdm.auto import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RecursiveGAN(torch.nn.Module):
    """
    Encoding + GAN

    The noise input to the Gen is taken as the output of an encoder
    """
    def __init__(self, z_size=100, dropout=0.5, batchnorm=True):
        super(RecursiveGAN, self).__init__()

        self.z_size = z_size
        self.gen = torch_gan.make_model_from_block(torch_gan.CNNTransposeBlock,
                                              z_dim=z_size,
                                              n_channels=[512, 256, 128, 128],
                                              kernel_sizes=[4, 4, 4, 4],
                                              strides=[1, 2, 2, 2],
                                              paddings=[0, 1, 1, 1],
                                              dropout=dropout,
                                              batchnorm=batchnorm)

        self.gen.add_module(name='output_conv', module=torch.nn.ConvTranspose2d(128, 3, 4, 2, 1))
        self.gen.add_module(name='output_activ', module=torch.nn.Sigmoid())

        ####
        self.disc = torch_gan.make_model_from_block(torch_gan.CNNBlock,
                                               z_dim=3,
                                               n_channels=[128, 128, 256, 512],
                                               kernel_sizes=[4, 4, 4, 4],
                                               strides=[2, 2, 2, 2],
                                               paddings=[1, 1, 1, 1],
                                               dropout=dropout,
                                               batchnorm=batchnorm)
        #self.disc.add_module(name='output_conv', module=torch.nn.Conv2d(512, 1, 4, stride=1))
        ###
        enc_hidden_size = 256
        rnd_in = torch.rand((1, 3, 64, 64))
        rnd_out = self.disc(rnd_in)
        print("RND out shape: %s" % str(rnd_out.shape))

        self.disc.add_module(name='z_flatten', module=Flatten())
        self.disc.add_module(name='z_linear', module=torch.nn.Linear(rnd_out.view(-1).shape[0],
                                                                      enc_hidden_size))
        #self.disc.add_module(name='z_bnorm', module=torch.nn.BatchNorm1d(z_size))
        self.disc.add_module(name='output_sigmoid', module=torch.nn.LeakyReLU())
        self.disc.add_module(name='output_linear', module=torch.nn.Linear(enc_hidden_size, 1))
        ####
        self.disc.add_module(name='output_activ', module=torch.nn.Sigmoid())

        ####
        #self.encoder = dict(self.disc.named_modules())['z_linear']
        #self.encoder = self.disc[:-2]
        #output = self.disc[:-1](rnd_in)
        self.encoder = nn.Sequential(
            self.disc[:-2],
            nn.Linear(enc_hidden_size, z_size),
            nn.BatchNorm1d(z_size),
            #nn.Tanh(),
        )
        self.disc.apply(weights_init)
        self.gen.apply(weights_init)
        self.encoder.apply(weights_init)


    def forward(self, x):
        #z = self.encoder(x.view(-1, self.input_size))
        z = self.encoder(x)
        g = self.gen(z.view(x.shape[0], -1, 1, 1))
        d = self.disc(g)
        return z, g, d

@attr.attrs
class RecursiveGAN_Trainer:
    model = attr.ib()
    data_gen = attr.ib()
    n_samples = attr.ib(None)
    gen_lr = attr.ib(0.0002)
    disc_lr = attr.ib(0.0002)
    enc_lr = attr.ib(0.0002)
    beta1 = attr.ib(0.5)
    criterion = attr.ib(torch.nn.BCELoss())
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    epochs_trained = attr.ib(0)

    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None, batch_cb_delta=3):
        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        real_label, fake_label = 1, 0
        real_labels, fake_labels = None, None

        #self.n_samples = self.n_samples if self.n_samples is not None else
        if self.n_samples is None:
            try:
                self.n_samples = len(self.data_gen)
            except:
                pass

        ####
        self.gen_model = self.model.gen.to(self.device)
        self.disc_model = self.model.disc.to(self.device)
        self.enc_model = self.model.encoder.to(self.device)

        z_size = self.model.z_size

        ####
        self.gen_optim = torch.optim.Adam(self.gen_model.parameters(), lr=self.gen_lr,
                                          betas=(self.beta1, 0.999))
        self.disc_optim = torch.optim.Adam(self.disc_model.parameters(), lr=self.disc_lr,
                                           betas=(self.beta1, 0.999))
        #self.enc_optim = torch.optim.Adam(self.enc_model[-2:].parameters(), lr=self.enc_lr)
        self.enc_optim = torch.optim.Adam(self.enc_model[-2:].parameters(), lr=self.enc_lr,
                                          betas=(self.beta1, 0.999))

        ####
        self.gen_losses = list()
        self.disc_losses = list()
        self.enc_losses = list()

        ###
        self.epoch_cb_history = [{k: cb(self, 0) for k, cb in epoch_callbacks.items()}]
        self.batch_cb_history = [{k: cb(self, 0) for k, cb in batch_callbacks.items()}]

        with tqdm(total=n_epochs, desc='Training epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                with tqdm(total=self.n_samples, desc='-loss-') as batch_pbar:
                    for i, data in enumerate(self.data_gen):
                        i += 1  # Epochs are not zero indexed

                        self.disc_model.zero_grad()

                        # Take a real batch
                        real_x = data[0].to(self.device)
                        batch_size = real_x.shape[0]
                        label_shape = (batch_size,)

                        ########
                        # Discriminator updating
                        ###
                        # Labels for the real batch in disc
                        if real_labels is None:
                            real_labels = torch.full(label_shape, real_label,
                                                     device=self.device).view(-1)

                        ####
                        # REAL Batch
                        disc_real_output = self.disc_model(real_x).view(-1)
                        d_real_err = self.criterion(disc_real_output, real_labels)
                        real_z = self.enc_model(real_x)
                        #real_z = torch.randn(batch_size, z_size, 1, 1, device=self.device)

                        #(d_real_err + enc_loss).backward()

                        ####
                        # Fake Batch
                        # Generator takes encodings as input
                        #real_z = self.enc_model(disc_intermediate_output)

                        # Generate images from encoded z
                        gen_real_z_output = self.gen_model(real_z.view(batch_size,
                                                                       z_size,
                                                                       1, 1))
                        # Disc binary classification of fake inputs
                        if fake_labels is None:
                            fake_labels = torch.full(label_shape,
                                                     fake_label, device=self.device).view(-1)

                        disc_fake_output = self.disc_model(gen_real_z_output).view(-1)
                        d_fake_err = self.criterion(disc_fake_output, fake_labels)
                        #d_fake_err.backward(retain_graph=True)
                        d_err = d_real_err + d_fake_err #+ enc_loss
                        d_err.backward(retain_graph=True)
                        #d_real_err.backward()
                        #d_fake_err.backward()

                        self.disc_optim.step()

                        #####
                        # Generator updating
                        ###
                        # Give noise to G, then check that it is labeled fake
                        self.gen_model.zero_grad()
                        self.enc_model.zero_grad()
                        real_z = self.enc_model(real_x)
                        #real_z = torch.randn(batch_size, z_size, 1, 1, device=self.device)
                        gen_real_z_output = self.gen_model(real_z.view(batch_size,
                                                                       z_size,
                                                                       1, 1))
                        disc_fake_output = self.disc_model(gen_real_z_output)

                        # Calculate an error for the generator (e.g. did we trick discrim)
                        g_d_err = self.criterion(disc_fake_output, real_labels)

                        BCE = torch.nn.functional.binary_cross_entropy(gen_real_z_output,
                                                                       real_x,
                                                                       reduction='mean')
                        #enc_loss = torch.norm(real_z) * 0.001
                        enc_loss = BCE
                        (g_d_err + BCE).backward(retain_graph=True)

                        self.gen_optim.step()
                        #self.enc_optim.step()

                        #####
                        # Encoder updating
                        ###

                        # Encourage small values of encoding
                        #z_l2 = torch.norm(real_z)
                        #kl_loss = torch.nn.functional.kl_div(real_z.abs().log(),
                        #                                       torch.rand_like(real_z, requires_grad=True),
                        #                                       reduction='batchmean')
                        #enc_loss = z_l2 + kl_loss

                        #enc_loss = torch.norm(real_z)
                        #enc_loss.backward()
                        #self.enc_optim.step()
                        #enc_loss = torch.tensor(0)


                        ####
                        # Record losses/diags
                        ###
                        self.gen_losses.append(g_d_err.item())
                        self.disc_losses.append(d_err.item())
                        self.enc_losses.append(enc_loss.item())


                        ####
                        # Description and progress updates
                        #desc_str = "Gen-L: %.3f || Disc-L: %.3f" % ()
                        desc_str = "Gen-L: %.3f || Disc-L: %.3f || Enc-L: %.3f" % (np.mean(self.gen_losses[-50:]),
                                                                                   np.mean(self.disc_losses[-50:]),
                                                                                   np.mean(self.enc_losses[-50:]))
                        batch_pbar.set_description(desc_str)
                        batch_pbar.update(1)

                        if not i % batch_cb_delta:
                            self.batch_cb_history.append({k: cb(self, epoch)
                                                          for k, cb in batch_callbacks.items()})

                self.epochs_trained += 1
                self.epoch_cb_history.append({k: cb(self, epoch) for k, cb in epoch_callbacks.items()})
                epoch_pbar.update(1)


