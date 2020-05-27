"""Top-level package for Deepgroove WEB user experiment."""

from flask import Flask

__author__ = """Fred Osterrath"""
__email__ = 'frederic.osterrath@mila.quebec'
__version__ = '0.1.0'

APP = Flask(__name__)

APP.secret_key = b"1qaz2wsx42!000077777"
