import numpy as np
import torch
from acgan import torch_gan, evaluation
from acgan import torch_vae
from torch import nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils

import attr
from tqdm.auto import tqdm

class EGAN(torch.nn.Module):
    def __init__(self, z_size=100, dropout=0.5, batchnorm=True):
        super(EGAN, self).__init__()

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
        self.disc.add_module(name='output_conv', module=torch.nn.Conv2d(512, 1, 4, stride=1))
        self.disc.add_module(name='output_activ', module=torch.nn.Sigmoid())

        ####
        #self.vae = torch_vae.VAE(input_size=(3*64*64))
        self.input_size = 3*64*64
        enc_hidden_size = 128
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, enc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(enc_hidden_size, z_size)
        )


    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_size))
        g = self.gen(z.view(x.shape[0], -1, 1, 1))
        d = self.disc(g)
        return z, g, d


@attr.attrs
class EGAN_Trainer:
    egan_model = attr.ib()
    data_gen = attr.ib()
    n_samples = attr.ib(None)
    learning_rate = attr.ib(0.0002)
    beta1 = attr.ib(0.5)
    criterion = attr.ib(torch.nn.BCELoss())
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    epochs_trained = attr.ib(0)

    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None, batch_cb_delta=3):
        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        real_label, fake_label = 1, 0

        #self.n_samples = self.n_samples if self.n_samples is not None else
        if self.n_samples is None:
            try:
                self.n_samples = len(self.data_gen)
            except:
                pass

        ####
        self.gen_model = self.egan_model.gen.to(self.device)
        self.disc_model = self.egan_model.disc.to(self.device)
        self.enc_model = self.egan_model.encoder.to(self.device)

        z_size = self.egan_model.z_size

        ####
        self.gen_optim = torch.optim.Adam(self.gen_model.parameters())
        self.disc_optim = torch.optim.Adam(self.disc_model.parameters())
        self.enc_optim = torch.optim.Adam(self.enc_model.parameters())

        ####
        self.gen_losses = list()
        self.disc_losses = list()
        self.enc_losses = list()

        ###
        epoch_cb_history = [{k: cb(self, 0) for k, cb in epoch_callbacks.items()}]
        batch_cb_history = [{k: cb(self, 0) for k, cb in batch_callbacks.items()}]

        with tqdm(total=n_epochs, desc='Training epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                with tqdm(total=self.n_samples, desc='-loss-') as batch_pbar:
                    for i, data in enumerate(self.data_gen):
                        self.disc_model.zero_grad()
                        self.gen_model.zero_grad()
                        self.enc_model.zero_grad()

                        # Take a real batch
                        real_x = data[0].to(self.device)
                        batch_size = real_x.shape[0]

                        ########
                        # Discriminator updating
                        ###

                        # Labels for the real batch in disc
                        real_labels = torch.full((batch_size, 1, 1, 1), real_label,
                                                 device=self.device)

                        # Discrims binary classification of real inputs
                        disc_real_output = self.disc_model(real_x).view(-1)
                        d_real_err = self.criterion(disc_real_output, real_labels)
                        d_real_err.backward()

                        # Generator takes encodings as input
                        real_z = self.enc_model(real_x.view(batch_size, -1))

                        # Generate images from encoded z
                        gen_real_z_output = self.gen_model(real_z.view(batch_size,
                                                                       z_size,
                                                                       1, 1))
                        # Disc binary classification of fake inputs
                        fake_labels = torch.full((batch_size, 1, 1, 1),
                                                 fake_label, device=self.device)
                        disc_fake_output = self.disc_model(gen_real_z_output)
                        d_fake_err = self.criterion(disc_fake_output, fake_labels)
                        d_fake_err.backward(retain_graph=True)
                        d_err = d_real_err + d_fake_err

                        self.disc_optim.step()

                        #####
                        # Generator updating
                        ###
                        # Discrim tries to classify fake images
                        disc_fake_output = self.disc_model(gen_real_z_output)

                        # Calculate an error for the generator (e.g. did we trick discrim)
                        g_d_err = self.criterion(disc_fake_output, real_labels)
                        g_d_err.backward(retain_graph=True)

                        self.gen_optim.step()

                        #####
                        # Encoder updating
                        ###

                        # Encourage small values of encoding
                        z_l2 = torch.norm(real_z)
                        z_l2.backward()

                        self.enc_optim.step()


                        ####
                        # Record losses/diags
                        ###
                        self.gen_losses.append(g_d_err.item())
                        self.disc_losses.append(d_err.item())
                        self.enc_losses.append(z_l2.item())


                        ####
                        # Description and progress updates
                        #desc_str = "Gen-L: %.3f || Disc-L: %.3f" % ()
                        desc_str = "Gen-L: %.3f || Disc-L: %.3f || Enc-L: %.3f" % (np.mean(self.gen_losses[-10:]),
                                                                                   np.mean(self.disc_losses[-10:]),
                                                                                   np.mean(self.enc_losses[-10:]))
                        batch_pbar.set_description(desc_str)
                        batch_pbar.update(1)


                self.epochs_trained += 1
                epoch_cb_history.append({k: cb(self, epoch) for k, cb in epoch_callbacks.items()})
                epoch_pbar.update(1)




if __name__ == """__main__""":
    dataloader = evaluation.make_celeba_dataloader("/export/datasets/celeba")
    egan = EGAN()
    egan_trainer = EGAN_Trainer(egan, dataloader)
    egan_trainer.train(3)
