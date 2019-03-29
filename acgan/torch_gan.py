import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils

import attr
from tqdm.auto import tqdm

def auto_extend(*args, max_len=None):
    args = [ list(a) if isinstance(a, (list, tuple)) else [a] for a in args]
    max_len = max(map(len, args)) if max_len is None else max_len
    args = [ a if len(a) == max_len else a + ([a[-1]] * (max_len - len(a))) for a in args]
    return list(zip(*args))

def make_model_from_block(cls, z_dim, n_channels, kernel_sizes,
                          strides, paddings, dilations=[1]):
    _iter = auto_extend(n_channels, kernel_sizes, strides,
                        paddings, #output_paddings,
                        dilations)
    model = torch.nn.Sequential()
    for i, (_n_chan, _kernel_size, _stride, _padding, _dilation) in enumerate(_iter):
        blk = cls(z_dim if not i else n_channels[i - 1],
                  _n_chan, kernel_size=_kernel_size, stride=_stride,
                  padding=_padding, #output_padding=_op,
                  groups=1, bias=True, dilation=_dilation)
        model.add_module(name="Block_%d" % i, module=blk)
    return model

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def intermediate_outputs(_m, input_data, show_layer=True, print_display=True):
    outputs = list()
    if print_display:
        print("Input Shape: %s" % str(tuple(input_data.shape)))
    for li in range(1, len(_m)+1):
        _out = _m[:li](input_data)
        if print_display:
            print("-- Layer %d --" % li)

        if print_display and show_layer:
            print(_m[li-1])

        if print_display:
            print(tuple(_out.shape))
            print("\t|\n\tV")

        outputs.append(_out)
    return outputs

class CNNTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    groups=1,
                    bias=True,
                    dilation=1):
        super(CNNTransposeBlock, self).__init__()

        self.convt = nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              output_padding=output_padding,
                              groups=groups,
                              bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.act = nn.ReLU(True)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.convt(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=True, batchnorm=True):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        #self.act = nn.ReLU(True)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.conv(x)
        if self.batchnorm:
            out = self.bn(out)
        out = self.act(out)
        return out

@attr.attrs
class GANTrainer():
    gen_model = attr.ib()
    disc_model = attr.ib()
    data_gen = attr.ib()
    in_channel_size = attr.ib(100)
    n_samples = attr.ib(None)
    learning_rate = attr.ib(0.0002)
    beta1 = attr.ib(0.5)
    criterion = attr.ib(torch.nn.BCELoss())
    disc_optim = attr.ib(None)
    gen_optim = attr.ib(None)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    epochs_trained = attr.ib(0, init=False)

    def init_weights(self, weight_func=weights_init):
        self.disc_model.apply(weight_func)
        self.gen_model.apply(weight_func)

    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None,
              batch_cb_delta=3):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks
        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Optimizers
        if self.disc_optim is None:
            self.disc_optim = torch.optim.Adam(self.disc_model.parameters(),
                                               lr=self.learning_rate,
                                               betas=(self.beta1, 0.999))
        if self.gen_optim is None:
            self.gen_optim = torch.optim.Adam(self.gen_model.parameters(),
                                              lr=self.learning_rate,
                                              betas=(self.beta1, 0.999))

        nz = self.in_channel_size

        self.epoch_losses = list()
        epoch_cb_history = [{k: cb(self, 0) for k, cb in epoch_callbacks.items()}]
        batch_cb_history = [{k: cb(self, 0) for k, cb in batch_callbacks.items()}]

        with tqdm(total=n_epochs,
                  desc='Training epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                G_losses = list()
                D_losses = list()
                with tqdm(total=self.n_samples, desc='-loss-') as batch_pbar:
                    for i, data in enumerate(self.data_gen):
                        self.disc_model.zero_grad()
                        # Format batch
                        real_cpu = data[0].to(self.device)
                        b_size = real_cpu.size(0)
                        label = torch.full((b_size,), real_label, device=self.device)
                        # Forward pass real batch through D
                        output = self.disc_model(real_cpu).view(-1)
                        # Calculate loss on all-real batch
                        errD_real = self.criterion(output, label)
                        # Calculate gradients for D in backward pass
                        errD_real.backward()
                        D_x = output.mean().item()

                        ## Train with all-fake batch
                        # Generate batch of latent vectors
                        noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                        # Generate fake image batch with G
                        fake = self.gen_model(noise)
                        label.fill_(fake_label)
                        # Classify all fake batch with D
                        output = self.disc_model(fake.detach()).view(-1)
                        # Calculate D's loss on the all-fake batch
                        errD_fake = self.criterion(output, label)
                        # Calculate the gradients for this batch
                        errD_fake.backward()
                        D_G_z1 = output.mean().item()
                        # Add the gradients from the all-real and all-fake batches
                        errD = errD_real + errD_fake
                        # Update D
                        self.disc_optim.step()

                        ############################
                        # (2) Update G network: maximize log(D(G(z)))
                        ###########################
                        self.gen_model.zero_grad()
                        label.fill_(real_label)  # fake labels are real for generator cost
                        # Since we just updated D, perform another forward pass of all-fake batch through D
                        output = self.disc_model(fake).view(-1)
                        # Calculate G's loss based on this output
                        errG = self.criterion(output, label)
                        # Calculate gradients for G
                        errG.backward()
                        D_G_z2 = output.mean().item()
                        # Update G
                        self.gen_optim.step()

                        # Save Losses for plotting later
                        G_losses.append(errG.item())
                        D_losses.append(errD.item())
                        batch_pbar.set_description("Gen-L: %.3f || Disc-L:%.3f" % (np.mean(G_losses[-20:]),
                                                                                        np.mean(D_losses[-20:])))
                        batch_pbar.update(1)
                        if not i%batch_cb_delta:
                            batch_cb_history.append({k: cb(self, epoch) for k, cb in batch_callbacks.items()})

                self.epoch_losses.append(dict(gen_losses=G_losses, disc_losses=D_losses))
                self.epochs_trained += 1
                epoch_cb_history.append({k: cb(self, epoch) for k, cb in epoch_callbacks.items()})
                epoch_pbar.update(1)
        return self.epoch_losses

    @staticmethod
    def grid_display(data, figsize=(10,10), pad_value=0, title=''):
        fig, ax = plt.subplots(figsize=figsize)

        #fig = plt.figure(figsize=figsize)
        ax.axis("off")
        ax.set_title(title)
        ax.imshow(np.transpose(vutils.make_grid(data,
                                                padding=2,
                                                normalize=True,
                                                pad_value=pad_value).detach().cpu(), (1, 2, 0)))

        return fig

    def display_batch(self, batch_size=16, noise=None, batch_data=None, figsize=(10, 10), title='Generated Data',
                      pad_value=0):
        if batch_data is None and noise is None:
            noise = torch.randn(batch_size, self.in_channel_size, 1, 1, device=self.device)
            fake_batch = self.gen_model(noise).to('cpu')
        elif batch_data is not None:
            fake_batch = batch_data
        else:
            # Generate fake image batch with G
            fake_batch = self.gen_model(noise).to('cpu')

        #real_batch = next(iter(self.data_gen))

        return self.grid_display(fake_batch, figsize=figsize, title=title, pad_value=pad_value)