from setuptools import find_packages, setup

setup(
    name="dormantspin",
    version="0.1",
    description="package for the simulation of ising-like models",
    author="godzilla-but-nicer",
    packages=find_packages(exclude=("tests",)),
)
