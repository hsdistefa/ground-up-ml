import os
import setuptools

__version__ = '0.0.1'

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(PROJECT_ROOT, 'requirements.txt')) as f:
    requirements = f.readlines()

setuptools.setup(
    name='groundupml',
    version=__version__,
    description='Explanatory machine learning from scratch in Python',
    author='Harrison DiStefano',
    url='https://github.com/hsdistefa/ground-up-ml',
    install_requires=requirements,
    setup_requires=['numpy>=1.10', 'scipy>=0.18'],
    packages=setuptools.find_packages()
)
