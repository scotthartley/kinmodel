"""Loads the default models defined in kinmodel.models in the
default_models dictionary.

"""
import pkgutil
import os
import importlib
import kinmodel.models

pkgpath = os.path.dirname(kinmodel.models.__file__)
filenames = [n for _, n, _ in pkgutil.iter_modules([pkgpath])]

default_models = {}
for file in filenames:
    module = importlib.import_module(f"kinmodel.models.{file}")
    default_models[module.model.name] = module.model
