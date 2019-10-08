from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='obstacle_tower_env',
    version='3.1',
    author='Unity Technologies',
    url='https://github.com/Unity-Technologies/obstacle-tower-env',
    py_modules=["obstacle_tower_env"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['mlagents-envs>=0.10,<0.11',
                      'gym']
)
