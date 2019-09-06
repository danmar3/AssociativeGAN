import pandas as pd
import numpy as np

import torch

import attr
from acgan import torch_gan
from acgan import torch_vae
from acgan.evaluate import nrds

import time


def torch_dcgan(dataloader, z_size=100, batchnorm=True, dropout=0.5):
    gen = torch_gan.make_model_from_block(torch_gan.CNNTransposeBlock,
                                           z_dim=z_size,
                                           n_channels=[512, 256, 128, 128],
                                           kernel_sizes=[4, 4, 4, 4],
                                           strides=[1, 2, 2, 2],
                                           paddings=[0, 1, 1, 1],
                                          dropout=dropout,
                                          batchnorm=batchnorm)

    gen.add_module(name='output_conv', module=torch.nn.ConvTranspose2d(128, 3, 4, 2, 1))
    gen.add_module(name='output_activ', module=torch.nn.Sigmoid())

    disc = torch_gan.make_model_from_block(torch_gan.CNNBlock,
                                           z_dim=3,
                                           n_channels=[128, 128, 256, 512],
                                           kernel_sizes=[4, 4, 4, 4],
                                           strides=[2, 2, 2, 2],
                                           paddings=[1, 1, 1, 1],
                                           dropout=dropout,
                                           batchnorm=batchnorm)
    disc.add_module(name='output_conv', module=torch.nn.Conv2d(512, 1, 4, stride=1))
    disc.add_module(name='output_activ', module=torch.nn.Sigmoid())

    trainer = torch_gan.GANTrainer(gen, disc, dataloader,
                                   n_samples=len(dataloader))#int(np.round(dataloader.sampler.num_samples / dataloader.batch_size)))

    return trainer

def torch_dense_vae(dataloader, z_size=100, hidden_size=1024):
    trainer = torch_vae.VAETrainer(model=torch_vae.VAE(input_size=64 ** 2,
                                                       hidden_size=hidden_size, z_size=z_size),
                                   data_gen=dataloader, n_samples=len(dataloader))
    return trainer



########
def make_celeba_dataloader(dataroot):
    import torchvision.transforms as transforms
    import torchvision.datasets as dset
    image_size = 64
    batch_size = 64
    workers = 4

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=torch.utils.data.RandomSampler(dataset,
                                                                                    replacement=True,
                                                                                    num_samples=2000*batch_size),
                                             shuffle=False, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloader


default_gan_trainer_funcs = {"torch_dscgan_%s_%s" % (str(bn), str(drp)):
                              lambda dataloader: torch_dcgan(dataloader, batchnorm=bn, dropout=drp)
                                  for bn in [True, False]
                                      for drp in [0., 0.25, 0.5]}

@attr.s
class Evaluation:
    dataloader = attr.ib()
    cooloff_time = attr.ib(60)  # For poor commodity machines
    gan_trainer_map = attr.ib(factory=(lambda: dict(default_gan_trainer_funcs)))


    def run_trainers(self, num_epochs=1, train_kws=None, post_train_hooks=None,
                     trainers_to_run=None):

        # Pull existing attributes if they exist to not overwrite various runs
        self.trainers = getattr(self, 'trainers', dict())
        self.trainer_results = getattr(self, 'trainer_results', dict())
        self.post_train_hook_results = getattr(self, 'post_train_hook_results',
                                               dict())
        self.trainer_latencies = getattr(self, 'trainer_latencies', dict())

        # Handle unset parameters
        trainers_to_run = (trainers_to_run if trainers_to_run is not None
                           else list(self.gan_trainer_map.keys()))
        train_kws = train_kws if train_kws is not None else dict()
        post_train_hooks = post_train_hooks if post_train_hooks is not None else list()

        # Train all models and store results for later
        for i, tname in enumerate(trainers_to_run):
            trainer_f = self.gan_trainer_map[tname]

            print("->Running[%d/%d]: %s" % (i+1, len(self.gan_trainer_map), str(tname)))
            trainer = trainer_f(dataloader=self.dataloader)
            start_t = time.time()
            res = trainer.train(num_epochs, **train_kws)
            self.trainer_latencies[tname] = time.time() - start_t

            self.trainer_results[tname] = res
            self.trainers[tname] = trainer
            self.post_train_hook_results[tname] = [h(trainer) for h in post_train_hooks]

            # Cool off
            if self.cooloff_time is not None:
                time.sleep(self.cooloff_time)

        return self.trainer_results

    def build_loss_frame(self):
        loss_frame = pd.DataFrame({m_name: {epoch: np.mean(loss_map['disc_losses'])
                                            for epoch, loss_map in enumerate(m_trainer.epoch_losses)}
                                   for m_name, m_trainer in self.trainers.items()})
        return loss_frame

    def compute_nrds(self):
        l_df = self.build_loss_frame()
        nrds_map = nrds.NRDS.score(**l_df.to_dict(orient='list'))
        return nrds_map

