import uuid
from typing import Annotated, Callable, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState, create_react_agent

from .base import AgentBase
from .message import MessageRefresh


class ReActAgent(AgentBase):

    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, description, model)
        self.tools = tools
        self.system_prompt = system_prompt

    def _build_graph(self) -> CompiledStateGraph:
        if self.tools:
            AgentState.__annotations__['messages'] = Annotated[list[AnyMessage], MessageRefresh(self.name).add_messages]
            compiled_graph = create_react_agent(model=self._model,
                                                tools=self.tools,
                                                state_schema=AgentState,
                                                state_modifier=self.system_prompt)
        else:
            prompt = self._create_prompt(self.system_prompt)
            agent = (prompt | self._model | StrOutputParser() |
                     RunnableLambda(lambda content: {'messages': [AIMessage(content=content, id=str(uuid.uuid4()))]}))
            MessagesState.__annotations__['messages'] = Annotated[list[AnyMessage],
                                                                  MessageRefresh(self.name).add_messages]

            graph = StateGraph(MessagesState)
            graph.add_node('agent', agent)
            graph.set_entry_point('agent')
            graph.set_finish_point('agent')
            compiled_graph = graph.compile()

        return compiled_graph
