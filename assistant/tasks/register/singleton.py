import sys
import threading
from typing import TypeVar

T = TypeVar('T')


class SingletonType(type):
    """
    Single-instance mode, ensuring that it is initialized only once
    """

    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with SingletonType._instance_lock:
                if not hasattr(cls, '_instance'):
                    cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


def make_singleton(obj_name: str, obj: T) -> T:
    """
    make sure singleton type for different import path
    """

    if obj_name not in sys.modules:
        sys.modules[obj_name] = obj  # noqa
    else:
        obj = sys.modules[obj_name]

    return obj
