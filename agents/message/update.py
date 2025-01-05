from langchain_core.messages import AnyMessage


def update_message(message: AnyMessage, **kwargs) -> AnyMessage:
    [setattr(message, key, value) if hasattr(message, key) else ... for key, value in kwargs.items()]
    return message
