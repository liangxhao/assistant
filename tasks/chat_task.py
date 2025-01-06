from ..agents import ReActAgent
from .base import TaskBase

SYSTEM_PROMPT = """

"""


class ChatTask(TaskBase):
    name: str = 'chat'
    description: str = '对话聊天专家，负责通用性的知识问答'

    def get_agent(self) -> ReActAgent:
        model = self._build_model(**self._model_config)
        agent = ReActAgent(name=self.name,
                           description=self.description,
                           model=model,
                           system_prompt=SYSTEM_PROMPT.strip())
        return agent
