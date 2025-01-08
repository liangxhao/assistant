from langchain_core.messages import AnyMessage
from langgraph.graph.message import Messages, add_messages


class MessageRefresh:

    def __init__(self, name: str):
        self._name = name

    def add_messages(self, left: Messages, right: Messages) -> Messages:
        left = add_messages(left, [])
        right = add_messages([], right)
        right = [self._update_message(item, name=self._name) for item in right]
        messages = add_messages(left, right)
        return messages

    def add_left_messages(self, left: Messages, right: Messages) -> Messages:
        return self.add_messages(left, [])

    def add_right_messages(self, left: Messages, right: Messages) -> Messages:
        return self.add_messages([], right)

    # noinspection PyMethodMayBeStatic
    def _update_message(self, message: AnyMessage, **kwargs) -> AnyMessage:
        [setattr(message, key, value) if hasattr(message, key) else ... for key, value in kwargs.items()]
        return message
