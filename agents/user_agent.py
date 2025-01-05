from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import MessagesState, StateGraph, add_messages

from .base.agent import BaseAgent


class UserAgent(BaseAgent):

    def __init__(
        self,
        name: str,
        description: str,
    ):
        super().__init__(name, description)

        agent = RunnableLambda(
            lambda state:
            {'messages': [HumanMessage(content=message.content, name=self.name)] for message in state['messages']})
        MessagesState.__annotations__['messages'] = Annotated[list[AnyMessage], lambda x, y: add_messages([], y)]

        graph = StateGraph(MessagesState)
        graph.add_node('agent', agent)
        graph.set_entry_point('agent')
        graph.set_finish_point('agent')
        self._graph = graph.compile()
