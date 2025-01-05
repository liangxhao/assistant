from typing import Annotated, Callable, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState, create_react_agent

from .base.agent import BaseAgent, create_prompt
from .message import ManageMessage


class ReActAgent(BaseAgent):

    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, description)
        if tools:
            AgentState.__annotations__['messages'] = Annotated[list[AnyMessage], ManageMessage(self.name).add_messages]
            self._graph = create_react_agent(model=model,
                                             tools=tools,
                                             state_schema=AgentState,
                                             state_modifier=system_prompt)
        else:
            prompt = create_prompt(system_prompt)
            agent = (prompt | model | StrOutputParser() |
                     RunnableLambda(lambda content: {'messages': [AIMessage(content=content)]}))
            MessagesState.__annotations__['messages'] = Annotated[list[AnyMessage],
                                                                  ManageMessage(self.name).add_messages]

            graph = StateGraph(MessagesState)
            graph.add_node('agent', agent)
            graph.set_entry_point('agent')
            graph.set_finish_point('agent')
            self._graph = graph.compile()
