from typing import List, Literal, Optional, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .base.agent import BaseAgent, create_prompt
from .state.principal import SummaryState
from .user_agent import UserAgent


class RouterAgent(BaseAgent):

    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel,
        agents: Sequence[BaseAgent],
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, description)

        user_agent = UserAgent(name='user', description="User")

        prompt = create_prompt(system_prompt)

        description_of_workers = '\n'.join([f"- {agent.name}: {agent.description}" for agent in agents])
        prompt = prompt.partial(description_of_workers=description_of_workers)

        workers: List[str] = [agent.name for agent in agents]

        class Router(TypedDict):
            """
            Worker to route to next. If no workers needed, route to FINISH.
            """

            next: Literal[*workers, 'FINISH']  # type: ignore

        model = prompt | model.with_structured_output(Router)

        def router(state: SummaryState) -> Command[Literal[*workers, '__end__']]:  # type: ignore
            response = model.invoke({'messages': state['messages']})
            goto = response['next']
            return Command(goto=END if goto == 'FINISH' else goto, update={'status': 'run'})


        def summary(state: SummaryState):
            messages = state['messages']
            router_name = messages[-1].name
            removed = []
            for message in messages[-2::-1]:
                if message.name == router_name:
                    removed.append(RemoveMessage(id=message.id))
                else:
                    break

            return {'messages': removed, "state": "run"}

        graph = StateGraph(SummaryState)
        graph.add_node(user_agent.name, user_agent.get_graph())
        graph.add_node('router', router)
        graph.add_node('summary', summary)
        graph.set_entry_point('user')
        graph.add_edge("user", "router")
        for agent in agents:
            graph.add_node(agent.name, agent.get_graph())
            graph.add_edge(agent.name, 'summary')
        graph.add_edge('summary', 'router')

        self._graph = graph.compile()
