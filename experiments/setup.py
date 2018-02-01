#!/usr/bin/env python
"""
distutils/setuptools install script.
"""
import re
from setuptools import find_packages, setup

VERSION_RE = re.compile(r'''([0-9dev.]+)''')


def get_version():
    with open('VERSION', 'rb') as fh:
        init = fh.read().strip()
    return VERSION_RE.search(init).group(1)


def get_requirements():
    with open('requirements.txt', 'rb') as f:
        return f.read().splitlines()


setup(
    name='winapi_deobf_experiments',
    version=get_version(),
    description='Windows API Deobfuscation Experiments',
    url='TBD',
    author='Michael T. Wojnowicz',
    author_email='mwojnowicz@cylance.com',
    package_dir={'': 'src/'},
    packages=find_packages('src/'),
    include_package_data=True,
    install_requires=get_requirements(),
    license="Cylance, Inc.",
)