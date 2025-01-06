import uuid
from typing import Callable, Literal, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .base import AgentBase


class ReflexAgent(AgentBase):

    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel,
        tools: Sequence[Union[BaseTool, Callable]],
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, description, model)
        self.tools = tools
        self.system_prompt = system_prompt

    def _build_graph(self) -> CompiledStateGraph:
        prompt = self._create_prompt(self.system_prompt)

        agent = (prompt | self._model | StrOutputParser() |
                 RunnableLambda(lambda content: {'messages': [AIMessage(content=content, id=str(uuid.uuid4()))]}))
        tools = [StructuredTool.from_function(tool) for tool in self.tools]
        assert len(tools) >= 1
        assert set(tools[0].args.keys()) == {'request', 'response'}

        tools = RunnableSequence(*tools) if len(tools) > 1 else tools[0]

        def _call_tools(state: MessagesState):
            assert isinstance(state['messages'][-1], AIMessage)
            assert isinstance(state['messages'][-2], HumanMessage)
            response = state['messages'][-1].content
            request = state['messages'][-2].content

            result = tools.invoke({'request': request, 'response': response})
            return {
                'messages': [
                    ToolMessage(
                        content=result['stdout'] or result['stderr'],
                        artifact=result,
                        status='success' if not result['stderr'] else 'error',
                        tool_call_id='tool',
                    )
                ]
            }

        def _gather(state: MessagesState):
            message = state['messages'][-1]
            assert isinstance(message, ToolMessage)
            return {'messages': [AIMessage(content=message.content)]}

        def _gather_should_end(state: MessagesState) -> Literal['agent', '__end__']:
            messages = state['messages']
            for message in messages[::-1]:
                if isinstance(message, ToolMessage):
                    return '__end__' if message.status == 'success' else 'agent'
            return 'agent'

        graph = StateGraph(MessagesState)
        graph.add_node('agent', agent)
        graph.add_node('tools', _call_tools)
        graph.add_node('gather', _gather)
        graph.add_edge('agent', 'tools')
        graph.add_edge('tools', 'gather')
        graph.add_conditional_edges('gather', _gather_should_end)
        graph.set_entry_point('agent')
        return graph.compile()
