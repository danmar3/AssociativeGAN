

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
    'celeb_a': {
        'dcgan': {
            'model': {
                'generator': {
                    'init_shape': (4, 4, 256),
                    'units': [1024, 512, 256, 128, 3],
                    'kernels': [(3, 3), (3, 3), (4, 4), (4, 4), (4, 4)],
                    'strides': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
                    'padding': ['same', 'same', 'same', 'same', 'same']
                },
                'discriminator': {
                    'units': [128, 256, 512, 1024],
                    'kernels': 4,
                    'strides': 2,
                    'dropout': None}
                },
            'train': {
                'gen/steps': 1,
                'dis/steps': 5
                }
            }
        }
    }
