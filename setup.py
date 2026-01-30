#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(packages=find_packages(where='src'))