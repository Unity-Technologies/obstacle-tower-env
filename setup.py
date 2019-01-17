from setuptools import setup, find_packages
from os import path

setup(
    name='obstacle_tower_env',
    version='0.1',
    py_modules=["obstacle_tower_env"],
    install_requires=['mlagents==0.6', 'gym-unity==0.1.1'],
)
