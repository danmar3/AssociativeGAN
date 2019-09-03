try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze
from setuptools import setup, find_packages
# for development installation: pip install -e .
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/project_name.tar.gz
DEPS = ['tensorflow-probability==0.6.0',
        'tensorflow-gpu==1.13.1',
        'tensorflow-datasets',
        'pandas', 'matplotlib', 'jupyter', 'scipy',
        'attrs']


def get_dependencies():
    tf_names = ['tensorflow-gpu', 'tensorflow', 'tf-nightly']
    tf_installed = any([any(tfname == installed.split('==')[0]
                            for tfname in tf_names)
                        for installed in freeze.freeze()])
    if tf_installed:
        return DEPS
    else:
        tf_in_deps = any([any(tfname == dep.split('==')[0]
                              for tfname in tf_names)
                          for dep in DEPS])
        if tf_in_deps:
            return DEPS
        else:
            return DEPS + ['tensorflow']


setup(name='acgan',
      version='0.1',
      packages=find_packages(
          exclude=["*test*", "tests"]),
      package_data={'': ['*.so']},
      install_requires=get_dependencies(),
      author='Daniel L. Marino',
      author_email='marinodl@vcu.edu',
      licence='GPL',
      )
