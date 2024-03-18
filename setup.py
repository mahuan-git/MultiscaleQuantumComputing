from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'pyscf',
    'matplotlib',
    'numpy'
]


setup(name='mqc',
      version='0.1',
      description='Multiscale quantum computing package for quantum computing chemistry',
      author='Ma Huan',
      author_email='mahuan15@mail.ustc.edu.cn',
      packages=find_packages(),
      install_requires=REQUIRED_PACKAGES,
      platforms=['any'],
     )