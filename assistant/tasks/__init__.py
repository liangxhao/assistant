from .register import task_table

__all__ = ['task_table']

import importlib
import os
import pkgutil

for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    importlib.import_module(f".{module_name}", package=__name__)
