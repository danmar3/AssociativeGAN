

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
            'model': {
                'embedding_size': 128,
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
                }
            }
        }
    }
