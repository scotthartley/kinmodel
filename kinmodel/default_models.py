"""Loads the default models defined in kinmodel.models in the 
default_models dictionary.

"""
import pkgutil, os, importlib
from . import models

pkgpath = os.path.dirname(models.__file__)
filenames = [n for _, n, _ in pkgutil.iter_modules([pkgpath])]

default_models = {}
for file in filenames:
    module = importlib.import_module(f"kinmodel.models.{file}")
    default_models[module.model.name] = module.model
