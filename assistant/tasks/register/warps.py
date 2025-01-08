import inspect
import os

from ..base.task import TaskBase
from .task_table import task_table

__REGISTERED_PATHS = set()


def __has_registered(obj):
    filepath = os.path.normpath(inspect.getsourcefile(obj))
    if filepath in __REGISTERED_PATHS:
        return True
    else:
        __REGISTERED_PATHS.add(filepath)
        return False


def register_task():
    """
    decorator for task
    """

    def register(cls):
        if not issubclass(cls, TaskBase):
            raise ValueError(f"{cls.__name__} is not the sub class of TaskBase in file "
                             f"{os.path.normpath(inspect.getsourcefile(cls))}.")

        if not __has_registered(cls):
            task_table.add_task(cls.name, cls)
        return cls

    return register
