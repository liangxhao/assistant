from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI

from ...agents import AgentBase
from .model import build_openai_model
from .setting import config


class TaskBase(ABC):
    name: str
    description: str

    def __init__(self):
        assert self.name and isinstance(self.name, str), f"The task name must be string for {self.__class__.__name__}"
        assert self.description and isinstance(
            self.description, str), f"The task description must be string for {self.__class__.__name__}"

        self._agent: AgentBase = self._build_agent()

    def _build_model(self) -> ChatOpenAI:
        model_config = config.get('model')
        return build_openai_model(**model_config.get(self.name, model_config.get('default')))

    @property
    def agent(self) -> AgentBase:
        return self._agent

    @abstractmethod
    def _build_agent(self) -> AgentBase:
        pass

    @abstractmethod
    def _point(self) -> str:
        pass

    @property
    def entry_point(self) -> str:
        pass

    @property
    def finish_point(self) -> str:
        pass
