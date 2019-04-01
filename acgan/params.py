

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
                    'dropout': 0.3}
                },
            'train': {
                'generator': {'regularizer': {'scale': 0.001}},
                'discriminator': {'regularizer': {'scale': 0.001}}
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
            }
        }
    }
