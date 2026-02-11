from setuptools import setup, find_packages

setup(
    name='chargequbit',
    version='0.1.0',
    packages=find_packages(include=['src']), 
    install_requires=[
        'matplotlib>=3.10.8',
        'numpy>=2.4.2',
        'scipy>=1.17.0',
        'Shapely>=2.1.2',
        'scikit-image>=0.0',
        'tabulate>=0.9.0',
        'zeroheliumkit>=0.5.5'
    ]
)