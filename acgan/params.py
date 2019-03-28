

PARAMS = {
    'mnist': {
        'dcgan': {
            'model': {
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
            }
        },
    'celeb_a_full': {
        'dcgan': {
            'model': {
                'generator': {
                    'init_shape': (7, 6, 256),
                    'units': [512, 256, 128, 64, 3],
                    'kernels': [(5, 4), (5, 4), (5, 5), (5, 5), (5, 5)],
                    'strides': [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)],
                    'padding': ['valid', 'valid', 'valid', 'valid', 'same']
                },
                'discriminator': {
                    'units': [64, 128],
                    'kernels': 5,
                    'strides': 2,
                    'dropout': 0.3}
                },
            'train': {
                'gen/steps': 1,
                'dis/steps': 5
                }
            }
        }
    }
