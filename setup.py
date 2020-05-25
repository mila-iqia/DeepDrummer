#!/usr/bin/env python

from distutils.core import setup

setup(
    name='deepdrummer',
    version='1.0',
    packages=[
        'deepdrummer',
    ],
    install_requires=[
        'numpy',
        'torch',
        'torchaudio',
        'six',
        'soundfile',
        'tkinter',
        'sounddevice'
        'scipy',
        'matplotlib'
    ]
)
