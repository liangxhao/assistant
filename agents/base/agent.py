from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph


class AgentBase(ABC):

    def __init__(self, name: str, description: str, model: Optional[BaseChatModel] = None):
        self._name: str = name
        self._description: str = description
        self._model: BaseChatModel = model

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def graph(self) -> CompiledStateGraph:
        return self._build_graph()

    # noinspection PyMethodMayBeStatic
    def _create_prompt(self, system_prompt: Optional[str] = None) -> ChatPromptTemplate:
        prompt = (ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            ('placeholder', '{messages}'),
        ]) if system_prompt else ChatPromptTemplate.from_messages([
            ('placeholder', '{messages}'),
        ]))
        return prompt
