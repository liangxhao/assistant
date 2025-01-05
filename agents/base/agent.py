from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph


def create_prompt(system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    prompt = (ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('placeholder', '{messages}'),
    ]) if system_prompt else ChatPromptTemplate.from_messages([
        ('placeholder', '{messages}'),
    ]))
    return prompt


class BaseAgent(ABC):

    @abstractmethod
    def __init__(self, name: str, description: str):
        self._name: str = name
        self._description: str = description
        self._graph: Optional[CompiledStateGraph] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def get_graph(self) -> CompiledStateGraph:
        assert self._graph
        return self._graph
