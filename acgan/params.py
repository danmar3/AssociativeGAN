import copy

PARAMS = {
    'mnist': {
        'dcgan': {
            'model': {
                'embedding_size': 128,
                'generator': {
                    'init_shape': (7, 7, 256),
                    'units': [128, 64, 1],
                    'kernels': 5,
                    'strides': [2, 2, 1]},
                'discriminator': {
                    'units': [64, 128],
                    'kernels': 5,
                    'strides': 2,
                    'dropout': 0.3}
                },
            'train': {
                'gen/steps': 1,
                'dis/steps': 1
                }
            },
        'msg-gan': {
            'model': {
                'embedding_size': 128,
                'generator': {
                    'init_shape': (4, 4, 128),
                    'units': [64, 32, 1],
                    'kernels': 3,
                    'strides': [2, 2, 2],
                    'padding': ['same', 'same', 'same', 'same']},
                'discriminator': {
                    'units': [32, 64, 128],
                    'kernels': 3,
                    'strides': [2, 2, 2],
                    'dropout': 0.1}
                },
            'train': {
                'generator': {'regularizer': {'scale': 1e-5}},
                'discriminator': {'regularizer': {'scale': 1e-5}}
                }
            }
        },
    'celeb_a': {
        'dcgan': {
            'model': {
                'embedding_size': 256,
                'generator': {
                    'init_shape': (4, 4, 1024),
                    'units': [512, 256, 128, 3],
                    'kernels': [(5, 5), (5, 5), (5, 5), (5, 5)],
                    'strides': [[2, 2], [2, 2], [2, 2], [2, 2]],
                    'padding': ['same', 'same', 'same', 'same']
                    },
                'discriminator': {
                    'units': [128, 256, 512, 1024],
                    'kernels': 5,
                    'strides': 2,
                    'dropout': None
                    }
                },
            'train': {
                'gen/steps': 1,
                'dis/steps': 5
                }
            },
        'gmmgan': {
            'global': {
                'USE_BIAS': {'generator': True, 'discriminator': True},
                'USE_BATCH_NORM': False,
                'GeneratorFeatureNorm': 'vector',
                'GeneratorGlobal': {'fnorm_hidden': True}
                },
            'data': {},
            'model': {
                'embedding_size': 64,
                'embedding': {'n_components': 100, 'min_scale_p': 0.1,
                              'constrained_loc': True
                              },
                'encoder': {
                    'units': [32, 64, 64, 64],
                    'kernels': 3,
                    'strides': 1,
                    'pooling': 2},
                'generator': {
                    'init_shape': (4, 4, 512),
                    'units': [512, 512, 512, 256, 128],
                    'add_noise': [False, True, True, True, False],
                    'outputs': 3,
                    'kernels': 3,
                    'strides': 2},
                'discriminator': {
                    'units': [128, 256, 512, 512, 512],
                    'kernels': 3,
                    'strides': 2,
                    'dropout': None}
                },
            'generator_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'discriminator_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                'loss_type': 'simplegp'
                },
            'encoder_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                'loss': {'embedding_kl': 0.005, 'use_zsim': True,
                         'comp_loss': 'kl5'}
                },
            'run': {
                'gan_steps': 200,
                'encoder_steps': 200,
                'embedding_steps': 50,
                'homogenize': False,
                'reset_embedding': 5}
            }
        },
    'celeb_a_hd_512': {
        'gmmgan': {
            'model': {
                'embedding_size': 128,
                'embedding': {'n_components': 100},
                'encoder': {
                    'units': [32, 64, 64, 64, 64],
                    'kernels': 3,
                    'strides': 1,
                    'pooling': 2},
                'generator': {
                    'init_shape': (4, 4, 512),
                    'units': [512, 512, 512, 256, 128, 64, 64],
                    'outputs': 3,
                    'kernels': 3,
                    'strides': 2},
                'discriminator': {
                    'units': [64, 64, 128, 256, 512, 512, 512],
                    'kernels': 3,
                    'strides': 2,
                    'dropout': None}
                },
            'generator_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'discriminator_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'encoder_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                'loss': {'embedding_kl': 0.001}
                },
            'run': {
                'gan_steps': 200,
                'encoder_steps': 200,
                'embedding_steps': 20,
                }
            }
        },
    'imagenet_512': {
        'gmmgan': {
            'model': {
                'embedding_size': 128,
                'embedding': {'n_components': 100},
                'encoder': {
                    'units': [32, 64, 64, 64, 64],
                    'kernels': 3,
                    'strides': 1,
                    'pooling': 2},
                'generator': {
                    'init_shape': (4, 4, 512),
                    'units': [512, 512, 256, 256, 128, 64, 64],
                    'outputs': 3,
                    'kernels': 3,
                    'strides': 2},
                'discriminator': {
                    'units': [64, 64, 128, 256, 256, 512, 512],
                    'kernels': 3,
                    'strides': 2,
                    'dropout': None}
                },
            'generator_trainer': {
                'batch_size': 4,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'discriminator_trainer': {
                'batch_size': 4,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'encoder_trainer': {
                'batch_size': 4,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                'loss': {'embedding_kl': 0.001}
                },
            'run': {
                'gan_steps': 200,
                'encoder_steps': 200,
                'embedding_steps': 20,
                }
            }
        },
    'imagenet_256': {
        'gmmgan': {
            'model': {
                'embedding_size': 128,
                'embedding': {'n_components': 100},
                'encoder': {
                    'units': [32, 64, 64, 64, 64],
                    'kernels': 3,
                    'strides': 1,
                    'pooling': 2},
                'generator': {
                    'init_shape': (4, 4, 512),
                    'units': [512, 512, 512, 256, 128, 64],
                    'outputs': 3,
                    'kernels': 3,
                    'strides': 2},
                'discriminator': {
                    'units': [64, 128, 256, 512, 512, 512],
                    'kernels': 3,
                    'strides': 2,
                    'dropout': None}
                },
            'generator_trainer': {
                'batch_size': 8,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'discriminator_trainer': {
                'batch_size': 8,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'encoder_trainer': {
                'batch_size': 8,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                'loss': {'embedding_kl': 0.01, 'use_zsim': True}
                },
            'run': {
                'gan_steps': 200,
                'encoder_steps': 200,
                'embedding_steps': 20,
                }
            }
        },
    # python3 gmmgan_test.py --n_steps=100 --n_steps_save=3 --gpu=7 --dataset="imagenet_128" --datadir="/home/marinodl/datasets/"
    'imagenet_128': {
        'gmmgan': {
            'global': {
                'USE_BIAS': {'generator': False, 'discriminator': False},
                'USE_BATCH_NORM': False},
            'model': {
                'embedding_size': 512,
                'embedding': {'n_components': 100},
                'encoder': {
                    'units': [32, 64, 64, 64],
                    'kernels': 3,
                    'strides': 1,
                    'pooling': 2},
                'generator': {
                    'init_shape': (4, 4, 512),
                    'units': [512, 512, 512, 256, 128],
                    'outputs': 3,
                    'kernels': 3,
                    'strides': 2},
                'discriminator': {
                    'units': [128, 256, 512, 512, 512],
                    'kernels': 3,
                    'strides': 2,
                    'dropout': None}
                },
            'generator_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'discriminator_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                },
            'encoder_trainer': {
                'batch_size': 16,
                'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
                'loss': {'embedding_kl': 0.001, 'use_zsim': True,
                         'comp_loss': 'kl5'}
                },
            'run': {
                'gan_steps': 200,
                'encoder_steps': 200,
                'embedding_steps': 50,
                'homogenize': True,
                'reset_embedding': 5,
                'n_start': 100
                }
            }
        },
    }


PARAMS['rockps'] = {'gmmgan': copy.deepcopy(PARAMS['celeb_a']['gmmgan'])}
PARAMS['rockps']['gmmgan']['model'] = {
    'embedding_size': 32,
    'embedding': {'n_components': 20, 'min_scale_p': 0.1},
    'encoder': {
        'units': [32, 64, 64, 64],
        'kernels': 3,
        'strides': 1,
        'pooling': 2},
    'generator': {
        'init_shape': (4, 4, 512),
        'units': [512, 512, 256, 256, 128],
        'outputs': 3,
        'kernels': 3,
        'strides': 2},
    'discriminator': {
        'units': [128, 256, 256, 512, 512],
        'kernels': 3,
        'strides': 2,
        'dropout': None}
    }

# python3 gmmgan_test.py --n_steps=100 --n_steps_save=5 --gpu=7 --dataset="cats_vs_dogs"
PARAMS['cats_vs_dogs'] = {'gmmgan': copy.deepcopy(PARAMS['celeb_a']['gmmgan'])}
PARAMS['cats_vs_dogs']['gmmgan']['encoder_trainer'] = {
    'batch_size': 16,
    'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
    'loss': {'embedding_kl': 0.1, 'use_zsim': True}
    }
PARAMS['cats_vs_dogs']['gmmgan']['model'] = {
    'embedding_size': 128,
    'embedding': {'n_components': 20, 'min_scale_p': 0.1},
    'encoder': {
        'units': [32, 64, 64, 128],
        'kernels': 3,
        'strides': 1,
        'pooling': 2},
    'generator': {
        'init_shape': (4, 4, 512),
        'units': [512, 512, 512, 256, 128],
        'outputs': 3,
        'kernels': 3,
        'strides': 2},
    'discriminator': {
        'units': [128, 256, 512, 512, 512],
        'kernels': 3,
        'strides': 2,
        'dropout': None}
    }


# python3 gmmgan_test.py --n_steps=100 --n_steps_save=5 --gpu=7 --dataset="stanford_dogs"
PARAMS['stanford_dogs'] = {
    'gmmgan': copy.deepcopy(PARAMS['celeb_a']['gmmgan'])
    }
PARAMS['stanford_dogs']['gmmgan']['global'] = {
    'USE_BIAS': {'generator': True, 'discriminator': True},
    'USE_BATCH_NORM': False,
    'GeneratorFeatureNorm': 'vector',
    'GeneratorGlobal': {'fnorm_hidden': True}
    }
PARAMS['stanford_dogs']['gmmgan']['model'] = {
    'embedding_size': 256,
    'embedding': {'n_components': 50, 'min_scale_p': 0.1},
    'encoder': {
        'units': [32, 64, 64, 64, 128],
        'kernels': 3,
        'strides': 1,
        'pooling': 2},
    'generator': {
        'init_shape': (4, 4, 512),
        'units': [512, 512, 512, 256, 128],
        'add_noise': [False, True, True, True, False],
        'outputs': 3,
        'kernels': 3,
        'strides': 2},
    'discriminator': {
        'units': [128, 256, 512, 512, 512],
        'kernels': 3,
        'strides': 2,
        'dropout': None}
    }
PARAMS['stanford_dogs']['gmmgan']['discriminator_trainer'] = {
    'batch_size': 16,
    'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
    'loss_type': 'simplegp'
    }
PARAMS['stanford_dogs']['gmmgan']['encoder_trainer'] = {
    'batch_size': 16,
    'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
    'loss': {'embedding_kl': 0.005, 'use_zsim': True,
             'comp_loss': 'kl5'}
    }
PARAMS['stanford_dogs']['gmmgan']['run'] = {
    'gan_steps': 200,
    'encoder_steps': 200,
    'embedding_steps': 50,
    'homogenize': True,
    'reset_embedding': 5,
    'n_start': 100
    }

PARAMS['stanford_dogs64'] = {
    'gmmgan': copy.deepcopy(PARAMS['stanford_dogs']['gmmgan'])
    }
PARAMS['stanford_dogs64']['gmmgan']['model'] = {
    'embedding_size': 64,
    'embedding': {'n_components': 50, 'min_scale_p': 0.1},
    'encoder': {
        'units': [32, 64, 64, 64],
        'kernels': 3,
        'strides': 1,
        'pooling': 2},
    'generator': {
        'init_shape': (4, 4, 128),
        'units': [512, 512, 512, 256],
        'add_noise': [False, False, True, True],
        'outputs': 3,
        'kernels': 3,
        'strides': 2},
    'discriminator': {
        'units': [256, 512, 512, 512],
        'kernels': 3,
        'strides': 2,
        'dropout': None}
    }
PARAMS['stanford_dogs64']['gmmgan']['global'] = {
    'USE_BIAS': {'generator': False, 'discriminator': False},
    'USE_BATCH_NORM': False,
    'GeneratorFeatureNorm': 'vector',
    'GeneratorGlobal': {'fnorm_hidden': True}
    }
PARAMS['stanford_dogs64']['gmmgan']['run'] = {
    'gan_steps': 200,
    'encoder_steps': 200,
    'embedding_steps': 50,
    'homogenize': True,
    'reset_embedding': 5,
    'n_start': 10
    }

for category in ['dog', 'cat', 'cow', 'sheep']:
    PARAMS['lsun_{}'.format(category)] = {
        'gmmgan': copy.deepcopy(PARAMS['stanford_dogs']['gmmgan'])
    }

for category in ['dog64', 'cat64', 'cow64', 'sheep64']:
    PARAMS['lsun_{}'.format(category)] = {
        'gmmgan': copy.deepcopy(PARAMS['stanford_dogs64']['gmmgan'])
    }

PARAMS['stl10'] = {
    'gmmgan': copy.deepcopy(PARAMS['stanford_dogs']['gmmgan'])
    }
PARAMS['stl10_64'] = {
    'gmmgan': copy.deepcopy(PARAMS['stanford_dogs64']['gmmgan'])
    }

PARAMS['cifar10'] = {
    'gmmgan': copy.deepcopy(PARAMS['celeb_a']['gmmgan'])
    }
PARAMS['cifar10']['gmmgan']['global'] = {
    'USE_BIAS': {'generator': False, 'discriminator': False},
    'USE_BATCH_NORM': False
   }
PARAMS['cifar10']['gmmgan']['model'] = {
    'embedding_size': 128,
    'embedding': {'n_components': 50, 'min_scale_p': 0.1,
                  'constrained_loc': True},
    'encoder': {
        'units': [32, 64, 64],
        'kernels': 3,
        'strides': 1,
        'pooling': 2},
    'generator': {
        'init_shape': (4, 4, 512),
        'units': [512, 512, 256],
        'outputs': 3,
        'kernels': 3,
        'strides': 2},
    'discriminator': {
        'units': [256, 512, 512],
        'kernels': 3,
        'strides': 2,
        'dropout': None}
    }

PARAMS['cifar10']['gmmgan']['discriminator_trainer'] = {
    'batch_size': 16,
    'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
    'loss_type': 'simplegp'
    }
PARAMS['cifar10']['gmmgan']['encoder_trainer'] = {
    'batch_size': 16,
    'optimizer': {'learning_rate': 0.0005, 'beta1': 0.0},
    'loss': {'embedding_kl': 0.01, 'use_zsim': True,
             'comp_loss': 'kl5'}
    }
PARAMS['cifar10']['gmmgan']['run'] = {
    'gan_steps': 200,
    'encoder_steps': 200,
    'embedding_steps': 50,
    'homogenize': False,
    'reset_embedding': 5
    }


datasets = (['celeb_a', 'stanford_dogs', 'stanford_dogs64',
             'stl10', 'stl10_64',
             'cifar10'] +
            ['lsun_{}'.format(d) for d in ['dog', 'cat', 'cow', 'sheep']] +
            ['lsun_{}64'.format(d) for d in ['dog', 'cat', 'cow', 'sheep']]
            )

for dataset in datasets:
    PARAMS[dataset]['wacgan'] = copy.deepcopy(PARAMS[dataset]['gmmgan'])

for dataset in datasets:
    PARAMS[dataset]['wacganV2'] = copy.deepcopy(PARAMS[dataset]['wacgan'])

PARAMS['cifar10']['wacganV2']['model'] = {
    'embedding_size': 128,
    'embedding': {'n_components': 50, 'min_scale_p': 0.1,
                  'constrained_loc': True},
    'encoder': {
        'units': [64, 128, 256, 512],
        'kernels': 3,
        'strides': 1,
        'pooling': 2},
    'generator': {
        'init_shape': (4, 4, 512),
        'units': [512, 512, 256],
        'outputs': 3,
        'kernels': 3,
        'strides': 2},
    'discriminator': {
        'units': [256, 512, 512],
        'kernels': 3,
        'strides': 2,
        'dropout': None}
    }

PARAMS['cifar10']['wacganV2']['data'] = {
    'augment_data': True
}
