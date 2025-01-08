from ..agents import ReActAgent
from .base import TaskBase
from .register import register_task

SYSTEM_PROMPT = """
你是一个聊天小助手，名字叫做“小爱”
"""


@register_task()
class ChatTask(TaskBase):
    name: str = 'chat'
    description: str = '对话聊天专家，负责通用性的知识问答'

    def _build_agent(self) -> ReActAgent:
        model = self._build_model()
        agent = ReActAgent(name=self.name,
                           description=self.description,
                           model=model,
                           system_prompt=SYSTEM_PROMPT.strip())
        return agent
