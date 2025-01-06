from typing import List, Literal, Optional, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .base import AgentBase
from .state.principal import SummaryState


class RouterAgent(AgentBase):

    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel,
        agents: Sequence[AgentBase],
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, description, model)
        self.agents = agents
        self.system_prompt = system_prompt

    def _build_graph(self):
        prompt = self._create_prompt(self.system_prompt)

        description_of_workers = '\n'.join([f"- {agent.name}: {agent.description}" for agent in self.agents])
        prompt = prompt.partial(description_of_workers=description_of_workers)

        workers: List[str] = [agent.name for agent in self.agents]

        class Router(TypedDict):
            """
            Worker to route to next. If no workers needed, route to FINISH.
            """

            next: Literal[*workers, 'FINISH']  # type: ignore

        model = prompt | self._model.with_structured_output(Router, method='json_schema')

        def router(state: SummaryState) -> Command[Literal[*workers, '__end__']]:  # type: ignore
            response = model.invoke({'messages': state['messages']})
            goto = response['next']
            # assert all(
            #     message.name for message in state["messages"]
            # ), "All messages must have corresponding agent names."
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

            return {'messages': removed, 'state': 'run'}

        graph = StateGraph(SummaryState)
        graph.add_node('router', router)
        graph.add_node('summary', summary)
        graph.set_entry_point('router')
        for agent in self.agents:
            graph.add_node(agent.name, agent.graph)
            graph.add_edge(agent.name, 'summary')
        graph.set_finish_point('summary')

        return graph.compile()
