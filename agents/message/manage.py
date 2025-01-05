from langgraph.graph.message import Messages, add_messages

from .update import update_message


class ManageMessage:

    def __init__(self, name: str):
        self._name = name

    def add_messages(self, left: Messages, right: Messages) -> Messages:
        left = add_messages(left, [])
        right = add_messages([], right)
        right = [update_message(item, name=self._name) for item in right]
        messages = add_messages(left, right)
        return messages
