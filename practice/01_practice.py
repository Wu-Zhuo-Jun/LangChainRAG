# -*- coding: utf-8 -*-
"""
===========================================================================
练习：接入命令行并支持多次问询
===========================================================================
"""

import asyncio
import os
import re
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================================
# 环境配置
# ============================================================================
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
os.environ["DEEPSEEK_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"


@tool
def get_today_date() -> str:
    """
    查询当前当天日期，返回 YYYY-MM-DD 格式的日期字符串。
    """
    return datetime.now().strftime("%Y-%m-%d")


@tool
def calculate_expression(expression: str) -> str:
    """
    处理简易加减表达式，例如 1 + 2 或 5 - 3。

    Args:
        expression: 只包含两个数字和一个加号或减号的表达式
    """
    match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*([+-])\s*(-?\d+(?:\.\d+)?)\s*", expression
    )
    if not match:
        return "表达式格式不正确，请输入类似 1 + 2 或 5 - 3 的内容"

    left = float(match.group(1))
    operator = match.group(2)
    right = float(match.group(3))

    if operator == "+":
        result = left + right
    else:
        result = left - right

    return f"{left} {operator} {right} = {result}"


TOOLS = [get_today_date, calculate_expression]
TOOL_MAP = {tool_item.name: tool_item for tool_item in TOOLS}


async def create_model_with_tools():
    """创建并返回绑定了工具的聊天模型。"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未检测到 API Key。请先设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7,
    )
    return model.bind_tools(TOOLS)


def print_tool_calls(response: AIMessage) -> None:
    """打印模型是否决定调用工具，以及调用参数。"""
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        print("\n模型决定调用工具：")
        for call in response.tool_calls:
            print(f"  函数: {call['name']}")
            print(f"  参数: {call['args']}")
    else:
        print("\n模型本轮未调用工具。")


async def chat_once(model_with_tools, user_input: str) -> str:
    """处理单轮对话，必要时自动调用工具。"""
    messages = [
        HumanMessage(
            content=(
                "你是一个命令行助手。"
                "当用户询问今天日期时，调用 get_today_date。"
                "当用户输入类似 1 + 2、5 - 3 的简易表达式时，调用 calculate_expression。"
                "其余情况直接简洁回答。\n\n"
                f"用户输入：{user_input}"
            )
        )
    ]

    response = await model_with_tools.ainvoke(messages)
    print(response)
    messages.append(response)

    if isinstance(response, AIMessage):
        print_tool_calls(response)

    while isinstance(response, AIMessage) and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            selected_tool = TOOL_MAP[tool_name]
            tool_result = selected_tool.invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

    return (
        response.content if isinstance(response.content, str) else str(response.content)
    )


async def main():
    """启动命令行循环，支持多次问询。"""
    try:
        model_with_tools = await create_model_with_tools()
    except Exception as exc:
        print(f"模型初始化失败：{exc}")
        return

    print("智能助手已启动，支持查询日期和简易加减运算。")
    print("示例：今天几号？ / 1 + 2")
    print("输入 exit、quit 或 退出 可结束对话。")

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("助手：再见！")
            break

        try:
            answer = await chat_once(model_with_tools, user_input)
            print(f"助手：{answer}")
        except Exception as exc:
            print(f"助手：处理失败，错误信息：{exc}")


if __name__ == "__main__":
    asyncio.run(main())
