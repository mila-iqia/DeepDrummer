#!/usr/bin/env python

from distutils.core import setup

setup(
    name='deepdrummer',
    version='1.0',
    packages=[
        'deepdrummer',
        'deepdrummer.webserver',
    ],
    install_requires=[
        'numpy',
        'torch',
        'torchaudio',
        'six',
        'soundfile',
        'sounddevice',
        'scipy',
        'matplotlib'
    ]
)
