try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze
from setuptools import setup, find_packages
# for development installation: pip install -e .
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/project_name.tar.gz
DEPS = ['tensorflow-gpu', 'tensorflow-probability', 'pandas', 'twodlearn',
        'matplotlib', 'jupyter', 'scipy', 'tensorflow-datasets']


def get_dependencies():
    if any(['tensorflow' in installed for installed in freeze.freeze()]):
        return [dep for dep in DEPS if 'tensorflow' not in dep]
    else:
        return DEPS


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
