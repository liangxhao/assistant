from abc import ABC, abstractmethod
from typing import Any, Dict

import httpx
import yaml
from langchain_openai import ChatOpenAI

from ...agents import AgentBase


class TaskBase(ABC):
    name: str
    description: str

    def __init__(self, model_config_file: str):
        assert self.name, f"The task name cannot be empty for {self.__class__.__name__}"
        assert self.description, f"The task description cannot be empty for {self.__class__.__name__}"

        with open(model_config_file) as f:
            content: Dict[str, Any] = yaml.safe_load(f.read())
        self._model_config = content.get(self.name, content.get('common'))

    @abstractmethod
    def get_agent(self) -> AgentBase:
        pass

    @abstractmethod
    def get_entry_point(self) -> str:
        pass

    @abstractmethod
    def get_finish_point(self) -> str:
        pass

    # noinspection PyMethodMayBeStatic
    def _build_model(self, model: str, api_key: str, base_url: str, **kwargs: Dict[str, Any]):
        fields = set(ChatOpenAI.model_fields.keys())
        fields_kwargs = {k: v for k, v in kwargs.items() if k in fields}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in fields}
        model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(mounts={
                'http://': None,
                'https://': None
            }),
            model_kwargs=model_kwargs,
            **fields_kwargs,
        )
        return model
