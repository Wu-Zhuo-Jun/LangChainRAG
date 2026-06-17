# -*- coding: utf-8 -*-
"""
=============================================================================
Agent 模块
=============================================================================

负责创建和管理 Agent 实例。
"""

from typing import Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

import config
from llm import get_llm
from prompts import create_react_prompt
from tools import get_all_tools, get_tools_description
from memory import conversation as memory_module


# ============================================================================
# 全局 Agent 实例
# ============================================================================

_agent_executor: Optional[AgentExecutor] = None


def get_agent() -> AgentExecutor:
    """
    获取 Agent 执行器实例（单例）

    Returns:
        AgentExecutor 实例
    """
    global _agent_executor

    if _agent_executor is None:
        _agent_executor = _create_agent()

    return _agent_executor


def _create_agent() -> AgentExecutor:
    """
    创建 Agent 执行器

    Returns:
        配置好的 AgentExecutor
    """
    print("正在初始化 Agent...")

    # 获取 LLM
    llm = get_llm()

    # 获取工具
    tools = get_all_tools()
    print(f"已加载 {len(tools)} 个工具")

    # 创建 Prompt
    tools_description = get_tools_description()
    prompt = create_react_prompt(tools_description)

    # 创建 Agent
    # 使用 LangChain Hub 中的 ReAct prompt
    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm,
        tools,
        react_prompt,
    )

    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.VERBOSE,
        max_iterations=config.MAX_ITERATIONS,
        max_execution_time=config.MAX_EXECUTION_TIME,
        handle_parsing_errors=True,
        memory=memory_module.get_memory(),
    )

    print("Agent 初始化完成！")

    return agent_executor


def reset_agent():
    """
    重置 Agent 实例

    会清除对话历史
    """
    global _agent_executor
    _agent_executor = None
    memory_module.clear_history()
    print("Agent 已重置")


def run_agent(query: str) -> str:
    """
    运行 Agent 处理查询

    Args:
        query: 用户查询

    Returns:
        Agent 的回答
    """
    agent = get_agent()

    # 调用 Agent
    result = agent.invoke({"input": query})

    return result.get("output", "抱歉，我无法回答这个问题。")


async def run_agent_async(query: str) -> str:
    """
    异步运行 Agent 处理查询

    Args:
        query: 用户查询

    Returns:
        Agent 的回答
    """
    agent = get_agent()

    # 异步调用 Agent
    result = await agent.ainvoke({"input": query})

    return result.get("output", "抱歉，我无法回答这个问题。")


# ============================================================================
# 辅助函数
# ============================================================================


def get_agent_info() -> dict:
    """
    获取 Agent 信息

    Returns:
        包含 Agent 配置信息的字典
    """
    return {
        "max_iterations": config.MAX_ITERATIONS,
        "verbose": config.VERBOSE,
        "tools_count": len(get_all_tools()),
        "tools": [t.name for t in get_all_tools()],
    }
