from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class SummaryState(TypedDict):
    status: Literal['ask', 'reply', 'run']
    messages: Annotated[list[AnyMessage], add_messages]
    content: str
    artifact: Any
