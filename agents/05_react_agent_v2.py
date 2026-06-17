# -*- coding: utf-8 -*-
"""
=============================================================================
LangChain 新版 ReAct Agent 教程 (LangChain 1.3.x)
=============================================================================

ReAct = Reasoning + Acting

新版 LangChain 使用 LangGraph 架构，重新设计了 Agent 的实现方式。

核心循环：
1. Thought (思考) - 分析当前情况，决定下一步
2. Action (行动) - 执行工具
3. Observation (观察) - 获取结果
4. 重复直到完成任务

=============================================================================
"""

import os
from typing import List, Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool, BaseTool
from langchain.agents.factory import create_agent
from langchain_core.runnables import RunnableConfig


# ============================================================================
# 环境配置
# ============================================================================

os.environ["OPENAI_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ============================================================================
# 第一部分：ReAct 原理详解
# ============================================================================


def explain_react原理():
    """
    详解 ReAct 的工作原理
    """
    print("\n" + "=" * 60)
    print("【1. ReAct 原理详解】")
    print("=" * 60)

    print("""
ReAct 是一种结合推理和行动的 Agent 框架。

核心思想：
让 AI 不仅能思考（Reasoning），还能行动（Acting）。


┌─────────────────────────────────────────────────────────────┐
│                      ReAct 循环                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐                                               │
│   │  输入问题  │                                               │
│   └────┬─────┘                                               │
│        │                                                      │
│        ▼                                                      │
│   ┌──────────┐     ┌──────────────┐                         │
│   │ 思考思考  │────▶│ 决定行动方案  │                         │
│   │ Thought  │     │ Thought      │                         │
│   └────┬─────┘     └──────┬───────┘                         │
│        │                   │                                  │
│        │                   ▼                                  │
│        │            ┌──────────────┐                          │
│        │            │   执行行动    │                          │
│        │            │   Action    │                          │
│        │            └──────┬───────┘                          │
│        │                   │                                  │
│        │                   ▼                                  │
│        │            ┌──────────────┐                          │
│        │            │   观察结果    │                          │
│        │            │ Observation  │                          │
│        │            └──────┬───────┘                          │
│        │                   │                                  │
│        └───────────────────┼───────────────────────────────▶ 最终答案
│                            │                                   │
│                      循环继续？                                 │
└─────────────────────────────────────────────────────────────┘


ReAct Prompt 示例：

Question: 北京现在的天气如何？适合户外活动吗？

Thought: 我需要先查询北京现在的天气情况。
Action: get_weather
Action Input: {"location": "北京"}
Observation: 北京，晴，25°C，空气质量良好
Thought: 根据天气信息，北京天气晴朗，温度适宜，空气质量良好。
Final Answer: 北京今天天气晴朗，气温25°C，空气质量良好，非常适合户外活动。

""")

    input("\n按 Enter 继续...")


# ============================================================================
# 第二部分：定义工具
# ============================================================================


def create_tools() -> List[BaseTool]:
    """
    创建演示用的工具集合
    """

    @tool
    def get_weather(location: str) -> str:
        """查询指定城市的天气情况

        Args:
            location: 城市名称

        Returns:
            天气描述，包含温度、天气状况、空气质量等
        """
        weather_db = {
            "北京": {
                "weather": "晴",
                "temperature": "25°C",
                "air_quality": "良好",
                "humidity": "45%",
                "wind": "2-3级",
            },
            "上海": {
                "weather": "多云",
                "temperature": "28°C",
                "air_quality": "轻度污染",
                "humidity": "65%",
                "wind": "3-4级",
            },
            "广州": {
                "weather": "雷阵雨",
                "temperature": "32°C",
                "air_quality": "优",
                "humidity": "80%",
                "wind": "4-5级",
            },
            "深圳": {
                "weather": "晴",
                "temperature": "30°C",
                "air_quality": "良",
                "humidity": "55%",
                "wind": "2级",
            },
        }

        if location in weather_db:
            info = weather_db[location]
            return (
                f"{location}天气：{info['weather']}，"
                f"温度：{info['temperature']}，"
                f"空气质量：{info['air_quality']}，"
                f"湿度：{info['humidity']}，"
                f"风力：{info['wind']}"
            )
        else:
            return f"抱歉，暂不支持查询 {location} 的天气信息"

    @tool
    def calculate(expression: str) -> str:
        """执行数学计算

        Args:
            expression: 数学表达式，如 "2 + 3 * 4"、"100 / 5"

        Returns:
            计算结果
        """
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    @tool
    def search_news(keyword: str) -> str:
        """搜索新闻

        Args:
            keyword: 搜索关键词

        Returns:
            搜索结果摘要
        """
        news_db = {
            "AI": [
                "OpenAI 发布 GPT-5，性能大幅提升",
                "谷歌发布 Gemini 2.0，全面超越 GPT-4",
                "Claude 3 发布，支持多模态",
            ],
            "科技": [
                "苹果发布 iPhone 16 系列",
                "特斯拉全自动驾驶获批在中国运营",
                "SpaceX 完成第100次火箭回收",
            ],
            "经济": [
                "央行降准0.25个百分点",
                "比特币突破10万美元大关",
                "全球股市普遍上涨",
            ],
        }

        results = news_db.get(keyword, [])

        if results:
            return "\n".join([f"- {news}" for news in results])
        else:
            return f"暂无关于'{keyword}'的最新新闻"

    @tool
    def get_date_info() -> str:
        """获取当前日期和时间信息

        Returns:
            当前日期、星期信息
        """
        from datetime import datetime

        now = datetime.now()
        weekday_names = ["一", "二", "三", "四", "五", "六", "日"]

        return (
            f"今天是 {now.strftime('%Y年%m月%d日')}，"
            f"星期{weekday_names[now.weekday()]}，"
            f"时间 {now.strftime('%H:%M:%S')}"
        )

    return [get_weather, calculate, search_news, get_date_info]


# ============================================================================
# 第三部分：创建新版 ReAct Agent
# ============================================================================


async def demo_new_react_agent():
    """
    演示新版 LangGraph Agent 的创建和使用
    """
    print("\n" + "=" * 60)
    print("【2. 创建新版 ReAct Agent】")
    print("=" * 60)

    tools = create_tools()

    # 定义系统提示词
    system_prompt = """你是一个智能助手，名为小智。你能够使用工具来回答问题。

工作流程：
1. 理解用户的问题
2. 思考是否需要使用工具
3. 如果需要，调用合适的工具
4. 根据工具返回的结果给出回答

回复要求：
- 简洁明了
- 如果调用了工具，说明为什么调用
- 最终给出完整、有帮助的回答
"""

    # 使用新版 create_agent 创建 Agent
    print("\n--- 使用 create_agent 创建 Agent ---")
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    print(f"Agent 类型: {type(agent)}")

    print("\n" + "=" * 60)
    print("【3. 新版 ReAct Agent 执行示例】")
    print("=" * 60)

    # 示例1: 天气查询
    print("\n--- 示例1: 天气查询 ---")
    question = "北京今天的天气怎么样？适合出门吗？"
    print(f"问题: {question}")

    config: RunnableConfig = {"recursion_limit": 10}
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config)

    print("\n完整响应结构:")
    print(f"  - 类型: {type(result)}")
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            print(f"  - {type(msg).__name__}: {msg.content[:100]}...")

    # 提取最终回答
    if isinstance(result, dict) and "messages" in result:
        final_answer = result["messages"][-1].content
        print(f"\n最终答案: {final_answer}")

    # 示例2: 需要多步推理
    print("\n--- 示例2: 多步推理 ---")
    question2 = "北京和上海哪个城市更适合户外跑步？"
    print(f"问题: {question2}")

    result2 = await agent.ainvoke(
        {"messages": [HumanMessage(content=question2)]}, config
    )

    if isinstance(result2, dict) and "messages" in result2:
        final_answer2 = result2["messages"][-1].content
        print(f"\n最终答案: {final_answer2}")


# ============================================================================
# 第四部分：流式输出
# ============================================================================


async def demo_streaming():
    """
    演示 Agent 的流式输出
    """
    print("\n" + "=" * 60)
    print("【4. Agent 流式输出】")
    print("=" * 60)

    tools = create_tools()

    system_prompt = "你是一个乐于助人的智能助手。"

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    print("流式输出演示:")
    print("-" * 40)

    question = "什么是 AI？"
    print(f"问题: {question}")

    config: RunnableConfig = {"recursion_limit": 5}

    # 流式获取输出
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=question)]}, config
    ):
        print(f"收到 chunk: {type(chunk)}")
        if isinstance(chunk, dict):
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    print(f"  - {type(msg).__name__}: {msg.content[:50]}...")
            else:
                for key, value in chunk.items():
                    print(f"  - {key}: {str(value)[:50]}...")

    print("-" * 40)


# ============================================================================
# 第五部分：多轮对话
# ============================================================================


async def demo_conversation():
    """
    演示多轮对话
    """
    print("\n" + "=" * 60)
    print("【5. 多轮对话】")
    print("=" * 60)

    tools = create_tools()

    system_prompt = """你是一个友好的智能助手。请根据对话上下文回答问题。"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    config: RunnableConfig = {"recursion_limit": 10}

    # 对话历史
    messages = []

    questions = [
        "我叫小明，请问你叫什么名字？",
        "小明是谁？",
        "你能帮我查询北京的天气吗？",
    ]

    for q in questions:
        print(f"\n{'=' * 40}")
        print(f"用户: {q}")
        print("=" * 40)

        # 添加用户消息
        messages.append(HumanMessage(content=q))

        # 调用 Agent
        result = await agent.ainvoke({"messages": messages}, config)

        # 获取助手回复
        if isinstance(result, dict) and "messages" in result:
            assistant_msg = result["messages"][-1]
            print(f"\n助手: {assistant_msg.content}")

            # 将助手回复添加到历史
            messages.append(assistant_msg)


# ============================================================================
# 第六部分：错误处理
# ============================================================================


async def demo_error_handling():
    """
    演示错误处理
    """
    print("\n" + "=" * 60)
    print("【6. 错误处理】")
    print("=" * 60)

    tools = create_tools()

    system_prompt = "你是一个智能助手。"

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    # 设置较小的递归限制来测试错误处理
    config: RunnableConfig = {"recursion_limit": 3}

    test_cases = [
        "你好，你是谁？",  # 无需工具的问题
        "xyzabc123 是什么？",  # 无效查询
    ]

    for case in test_cases:
        print(f"\n测试: {case}")
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=case)]}, config
            )
            if isinstance(result, dict) and "messages" in result:
                print(f"结果: {result['messages'][-1].content}")
        except Exception as e:
            print(f"错误: {type(e).__name__}: {e}")


# ============================================================================
# 第七部分：自定义工具调用风格
# ============================================================================


async def demo_custom_style():
    """
    演示如何自定义工具调用风格
    """
    print("\n" + "=" * 60)
    print("【7. 自定义工具调用风格】")
    print("=" * 60)

    tools = create_tools()

    # 详细版本
    detailed_prompt = """你是一个专业的助手。

当你需要使用工具时：
1. 先思考：分析问题，确定需要什么信息
2. 再行动：选择合适的工具并执行
3. 最后总结：根据工具返回的信息给出答案

重要：
- 每次调用工具前，先解释为什么要调用
- 收到工具结果后，解释结果意味着什么
- 最终答案要完整、有条理
"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=detailed_prompt,
    )

    config: RunnableConfig = {"recursion_limit": 10}

    question = "帮我算一下 25 * 16 + 100 是多少？"
    print(f"问题: {question}")

    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config)

    print("\n完整响应结构:")
    print(f"  - 类型: {type(result)}")
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            print(f"  - {type(msg).__name__}: {msg.content[:100]}...")

    print("\n最终答案:")
    if isinstance(result, dict) and "messages" in result:
        print(f"\n答案: {result['messages'][-1].content}")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """
    运行所有演示
    """
    print("=" * 60)
    print("LangChain 新版 ReAct Agent 教程 (1.3.x)")
    print("=" * 60)

    # 1. 原理讲解
    explain_react原理()

    # 2. 新版 ReAct Agent
    # await demo_new_react_agent()

    # # 3. 流式输出
    # await demo_streaming()

    # # 4. 多轮对话
    # await demo_conversation()

    # # 5. 错误处理
    # await demo_error_handling()

    # 6. 自定义风格
    await demo_custom_style()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
