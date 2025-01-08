from typing import Any, Dict

import httpx
from langchain_openai import ChatOpenAI


# noinspection PyMethodMayBeStatic
def build_openai_model(self, model: str, api_key: str, base_url: str, **kwargs: Dict[str, Any]) -> ChatOpenAI:
    fields = set(ChatOpenAI.model_fields.keys())
    fields_kwargs = {k: v for k, v in kwargs.items() if k in fields}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in fields}
    model = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(mounts={
            'http://': None,
            'https://': None
        }),
        http_async_client=httpx.AsyncClient(mounts={
            'http://': None,
            'https://': None
        }),
        model_kwargs=model_kwargs,
        **fields_kwargs,
    )
    return model
