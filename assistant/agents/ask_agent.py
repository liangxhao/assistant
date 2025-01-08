from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import MessagesState, StateGraph

from .base import AgentBase
from .message import MessageRefresh


class AskAgent(AgentBase):

    def _build_graph(self):
        agent = RunnableLambda(
            lambda state: {'messages': [AIMessage(content=state['messages'][-1].content, id=state['messages'][-1].id)]})
        MessagesState.__annotations__['messages'] = Annotated[list[AnyMessage],
                                                              MessageRefresh(self.name).add_right_messages]

        graph = StateGraph(MessagesState)
        graph.add_node('agent', agent)
        graph.set_entry_point('agent')
        graph.set_finish_point('agent')
        return graph.compile()
