# -*- coding: utf-8 -*-
"""
=============================================================================
Prompt 模板
=============================================================================

定义 Agent 使用的各种 Prompt 模板。
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


# ============================================================================
# 系统提示词
# ============================================================================

SYSTEM_PROMPT = """你是一个智能助手，名字叫小智。你能够帮助用户回答问题、查询信息、执行计算等。

你的特点：
1. 友好、专业、有耐心
2. 回答简洁明了，重点突出
3. 如果不确定某个答案，会诚实说明
4. 能够使用工具来完成任务

请记住之前的对话内容，保持上下文连贯性。
"""


# ============================================================================
# 带记忆的对话模板
# ============================================================================

def create_conversation_prompt():
    """
    创建带对话历史的 Prompt 模板
    
    这个模板支持：
    1. 系统消息（设定角色）
    2. 对话历史（来自 Memory）
    3. 用户输入
    4. Agent 思考过程（scratchpad）
    
    Returns:
        ChatPromptTemplate 实例
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),  # 对话历史
        ("human", "{input}"),  # 用户输入
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent 思考过程
    ])


# ============================================================================
# ReAct Agent 模板
# ============================================================================

REACT_SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来回答问题。

你有以下工具可以使用：
{tools}

工作流程：
1. 分析用户问题，判断是否需要调用工具
2. 如果需要工具，选择合适的工具并提供参数
3. 根据工具返回结果，给出最终回答

重要：
- 每次只能调用一个工具
- 仔细检查工具参数是否正确
- 基于工具返回的真实结果回答用户
"""


def create_react_prompt(tools_description: str):
    """
    创建 ReAct 风格的 Prompt
    
    Args:
        tools_description: 工具描述字符串
    
    Returns:
        ChatPromptTemplate 实例
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=REACT_SYSTEM_PROMPT.format(tools=tools_description)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
