import os
import pathlib
from typing import Any, Dict

import yaml


def config_reader():
    root_path = pathlib.Path(__file__).resolve().parent.parent.parent
    env = os.getenv('ASSISTANT_ENV', 'DEV').lower()
    mapping = {'dev': 'dev.yaml', 'prod': 'prod.yaml'}

    filepath = root_path.joinpath('configs', mapping.get(env))
    with open(filepath) as f:
        config = yaml.safe_load(f)

    return config


config: Dict[str, Any] = config_reader()
