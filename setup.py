try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze
from setuptools import setup, find_packages
# for development installation: pip install -e .
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/project_name.tar.gz
DEPS = ['tensorflow-probability==0.6.0', 'tensorflow-datasets==1.2.0',
        'pandas', 'matplotlib', 'jupyter', 'scipy',
        'attrs', 'Pillow', 'GPUtil', 'scikit-learn',
        'opencv-python']

DEPS_DEV = ['Sphinx', 'sphinx_rtd_theme']


def get_dependencies():
    tf_names = ['tensorflow-gpu', 'tensorflow', 'tf-nightly']
    tf_installed = any([any(tfname == installed.split('==')[0]
                            for tfname in tf_names)
                        for installed in freeze.freeze()])
    if tf_installed:
        return DEPS
    else:
        return DEPS + ['tensorflow']


setup(name='acgan',
      version='0.1',
      packages=find_packages(
          exclude=["*test*", "tests"]),
      package_data={'': ['*.so']},
      install_requires=get_dependencies(),
      extras_require={
          'dev': DEPS_DEV
      },
      author='Daniel L. Marino',
      author_email='marinodl@vcu.edu',
      licence='GPL',
      )
