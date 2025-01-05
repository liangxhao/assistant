import logging

import langchain
from langchain_openai import ChatOpenAI

from agents import ReActAgent, ReflexAgent, RouterAgent
from prompts import router

# from utils.graph import show_graph

logging.basicConfig(level=logging.DEBUG)
langchain.debug = True
langchain.verbose = True

if __name__ == '__main__':
    # model = ChatOpenAI(model='Qwen2.5-7B-Instruct-GPTQ-Int8',
    #                    base_url='http://localhost:8080/v1',
    #                    api_key='empty',
    #                    temperature=0)
    model = ChatOpenAI(model='Qwen/Qwen2.5-72B-Instruct',
                       base_url='https://api-inference.modelscope.cn/v1',
                       api_key='be53b258-a6d8-431b-9d47-261ac2dfa328',
                       temperature=0)

    def tool1(request: str, response: str):
        """查询天气"""
        return '晴天'

    def tool2(x: float, y: float) -> float:
        """计算器，计算x + y"""
        return x + y

    def tool3(x: float, y: float) -> float:
        """计算器，计算x - y"""
        return x - y

    def tool4(x: float, y: float) -> float:
        """计算器，计算x * y"""
        return x * y

    def tool5(x: float, y: float) -> float:
        """计算器，计算x /y"""
        return x / y

    chat_worker = ReActAgent(name='chat', description='一般性的知识问题，不够专业', model=model)
    reflex_worker = ReflexAgent(name='weather', description='查询天气状况', model=model, tools=[tool1])
    react_worker = ReActAgent(
        name='calculator',
        description='计算器，专门处理各种四则运算任务',
        model=model,
        tools=[tool2, tool3, tool4, tool5],
    )

    router_worker = RouterAgent(
        name='router',
        description='路由专家，负责把用户的问题分发到不同人员',
        agents=[chat_worker, reflex_worker, react_worker],
        model=model,
        system_prompt=router.ROUTER_PROMPT,
    )

    # show_graph(router_worker.get_graph())

    inputs = {
        'messages': [(
            'user',
            # "你是谁",
            '帮我计算一下：1 + 2 - 3 等于多少',
        )],
    }

    events = router_worker.get_graph().stream(inputs, subgraphs=True, config={'recursion_limit': 50})
    for s in events:
        print(s)
        print('----')

    # result = router_worker.get_graph().invoke(inputs)
    # for i in result.get("messages"):
    #     i.pretty_print()
