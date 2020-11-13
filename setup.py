try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze
from setuptools import setup, find_packages
import importlib.util
# for development installation: pip install -e .
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/project_name.tar.gz
DEPS = ['pandas', 'matplotlib', 'jupyter', 'scipy',
        'attrs', 'Pillow', 'GPUtil', 'scikit-learn',
        'opencv-python', 'tensorflow-gan']

DEPS_DEV = ['Sphinx', 'sphinx_rtd_theme', 'jupyterlab', 'pdf2image', 'voila']


def get_dependencies():
    global DEPS

    def is_tensorflow(pkg_name):
        tf_names = ['tensorflow-gpu', 'tensorflow', 'tf-nightly']
        test1 = any(tfname == pkg_name.split('==')[0]
                    for tfname in tf_names)
        test2 = importlib.util.find_spec('tensorflow')
        return test1 or test2

    tf_installed = any([is_tensorflow(installed)
                        for installed in freeze.freeze()])
    if tf_installed:
        import tensorflow as tf
        version = tf.__version__
        if version not in ('1.13.1', '1.15.2', '1.15.3'):
            print('\n[twodlearn]: '
                  'ONLY 1.13.1 VERSION OF TENSORFLOW SUPPORTED!!!\n'
                  'Version 1.15 supported only if compiled locally.')
        if version == '1.13.1':
            DEPS = DEPS + [
                'tensorflow-probability==0.6.0', 'tensorflow-datasets==1.2.0']
        elif version in ('1.15.2', '1.15.3'):
            DEPS = DEPS + [
                'tensorflow-probability==0.8.0', 'tensorflow-datasets==3.2.1']
        else:
            DEPS = DEPS + ['tensorflow-probability', 'tensorflow-datasets']
        return DEPS
    else:
        return DEPS + ['tensorflow==1.13.1']


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
