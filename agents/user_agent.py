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
            {'messages': [HumanMessage(content=message.content, name=self.name) for message in state['messages']]})
        MessagesState.__annotations__['messages'] = Annotated[list[AnyMessage], lambda x, y: add_messages([], y)]

        graph = StateGraph(MessagesState)
        graph.add_node('agent', agent)
        graph.set_entry_point('agent')
        graph.set_finish_point('agent')
        self._graph = graph.compile()


if __name__ == '__main__':
    agent = UserAgent(name='user', description='user')
    # p = agent.get_graph().invoke({'messages': [HumanMessage(content='你好1'), HumanMessage(content='你好2')]}, config={'recursion_limit': 50})
    # print(p)

    inputs = {
        'messages': [(
            'user',
            # "你是谁",
            '帮我计算一下：1 + 2 - 3 等于多少',
        )],
    }

    events = agent.get_graph().stream(inputs, subgraphs=True, config={'recursion_limit': 50})
    for s in events:
        print(s)
        print('----')