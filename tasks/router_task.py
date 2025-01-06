from ..agents import RouterAgent
from .base import TaskBase

SYSTEM_PROMPT = """
You are a supervisor tasked with managing a conversation between the following workers:

{description_of_workers}

Given the following user request, respond with the worker to act next.
Each worker will perform a task and respond with their results and status.
When finished, respond with FINISH.
"""


class RouterTask(TaskBase):
    name: str = 'router'
    description: str = '任务路由专家，负责把任务分发给不同协作者'

    def get_agent(self) -> RouterAgent:
        model = self._build_model(**self._model_config)
        agent = RouterAgent(name=self.name,
                            description=self.description,
                            model=model,
                            agents=[],
                            system_prompt=SYSTEM_PROMPT.strip())
        return agent
