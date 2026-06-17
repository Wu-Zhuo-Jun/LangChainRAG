# -*- coding: utf-8 -*-
"""
=============================================================================
OpenAI 函数调用模式教程
=============================================================================

本教程讲解如何使用 OpenAI 的函数调用（Function Calling）功能：

1. 什么是 Function Calling
2. Tool 转 OpenAI 格式
3. 强制函数调用
4. 并行函数调用
5. 结构化输出

=============================================================================
"""

import os
import sys
import io

# 解决 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
from typing import List, Dict, Any, Optional, Union
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser


# ============================================================================
# 环境配置
# ============================================================================

os.environ["DEEPSEEK_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ============================================================================
# 第一部分：Function Calling 基础概念
# ============================================================================


def explain_function_calling():
    """
    解释 Function Calling 的概念
    """
    print("\n" + "=" * 60)
    print("【1. Function Calling 概念】")
    print("=" * 60)

    print("""
Function Calling（函数调用）是 OpenAI API 的一个强大功能：

传统方式：
┌──────────┐     用户输入      ┌──────────┐
│   用户   │ ───────────────▶ │   LLM    │ ──────▶ 文本回复
└──────────┘                  └──────────┘

Function Calling 方式：
┌──────────┐     用户输入      ┌──────────┐     工具调用      ┌──────────┐
│   用户   │ ───────────────▶ │   LLM    │ ───────────────▶ │  工具    │
└──────────┘                  └──────────┘                  └──────────┘
                               ◀───────────────
                               工具结果返回
                               ───────────────▶  最终回复
                               
                               
Function Calling 的优势：

1. 结构化输出：LLM 返回 JSON 格式的参数
2. 可靠性：避免 LLM 生成不稳定的 JSON
3. 可控性：强制调用特定函数
4. 工具集成：与外部系统（API、数据库）集成


OpenAI Function Calling 格式：

{
  "name": "get_weather",
  "description": "查询城市天气",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "城市名称"
      }
    },
    "required": ["location"]
  }
}

""")

    input("\n按 Enter 继续...")


# ============================================================================
# 第二部分：Tool 转 OpenAI 格式
# ============================================================================


def demo_convert_to_openai():
    """
    将 LangChain Tool 转换为 OpenAI 函数格式
    """
    print("\n" + "=" * 60)
    print("【2. Tool 转 OpenAI 格式】")
    print("=" * 60)

    # 定义一些 Tool
    @tool
    def get_weather(location: str, unit: str = "celsius") -> str:
        """查询指定城市的天气情况

        Args:
            location: 城市名称，如"北京"
            unit: 温度单位，可选 "celsius" 或 "fahrenheit"

        Returns:
            天气描述字符串
        """
        weather_db = {
            "北京": {"celsius": "晴，25°C", "fahrenheit": "晴，77°F"},
            "上海": {"celsius": "多云，28°C", "fahrenheit": "多云，82°F"},
        }

        city_weather = weather_db.get(location, {})
        return city_weather.get(unit, f"未找到 {location} 的天气数据")

    @tool
    def search_web(query: str, max_results: int = 5) -> str:
        """使用搜索引擎搜索信息

        Args:
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            搜索结果摘要
        """
        return f"关于 '{query}' 的搜索结果，共 {max_results} 条..."

    @tool
    def get_current_time() -> str:
        """获取当前时间

        Returns:
            当前日期和时间
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools = [get_weather, search_web, get_current_time]

    print("\n原始 LangChain Tools:")
    for t in tools:
        print(f"\n  名称: {t.name}")
        print(f"  描述: {t.description[:60]}...")
        print(f"  参数: {list(t.args.keys())}")

    # 转换为 OpenAI 格式
    print("\n" + "-" * 50)
    print("转换为 OpenAI Function Calling 格式:")

    openai_functions = [convert_to_openai_function(t) for t in tools]

    import json

    for i, func in enumerate(openai_functions):
        print(f"\n函数 {i + 1}: {func['name']}")
        print(json.dumps(func, indent=2, ensure_ascii=False)[:500] + "...")


# ============================================================================
# 第三部分：绑定函数到模型
# ============================================================================


async def demo_bind_functions():
    """
    将函数绑定到模型
    """
    print("\n" + "=" * 60)
    print("【3. 绑定函数到模型】")
    print("=" * 60)

    # 定义 Tool
    @tool
    def get_weather(location: str) -> str:
        """查询指定城市的天气"""
        return "晴，25°C"

    @tool
    def get_time() -> str:
        """获取当前时间"""
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")

    # 转换为 OpenAI 格式
    functions = [
        convert_to_openai_function(get_weather),
        convert_to_openai_function(get_time),
    ]

    # 绑定到模型
    model_with_functions = model.bind(
        functions=functions,
        function_call="auto",  # 自动决定是否调用函数
    )

    print("模型已绑定函数:")
    for f in functions:
        print(f"  - {f['name']}")

    # 测试查询
    print("\n--- 测试：正常对话 ---")
    response1 = await model_with_functions.ainvoke("你好，今天怎么样？")
    print(f"输入: 你好，今天怎么样？")
    print(f"响应: {response1.content}")
    if (
        hasattr(response1, "additional_kwargs")
        and "tool_calls" in response1.additional_kwargs
    ):
        print(f"函数调用: {response1.additional_kwargs['tool_calls']}")

    print("\n--- 测试：触发函数 ---")
    response2 = await model_with_functions.ainvoke("北京现在天气如何？")
    print(f"输入: 北京现在天气如何？")

    if hasattr(response2, "additional_kwargs"):
        tool_calls = response2.additional_kwargs.get("tool_calls", [])
        if tool_calls:
            print(f"触发的函数: {tool_calls[0]['function']['name']}")
            print(f"函数参数: {tool_calls[0]['function']['arguments']}")


# ============================================================================
# 第四部分：强制函数调用
# ============================================================================


async def demo_force_function_call():
    """
    强制模型调用特定函数
    """
    print("\n" + "=" * 60)
    print("【4. 强制函数调用】")
    print("=" * 60)

    @tool
    def get_weather(location: str) -> str:
        """查询天气"""
        return "晴，25°C"

    @tool
    def calculate(expression: str) -> str:
        """执行计算"""
        return str(eval(expression))

    functions = [
        convert_to_openai_function(get_weather),
        convert_to_openai_function(calculate),
    ]

    # 方式1: 强制调用某个函数
    print("\n方式1: 强制调用 calculate 函数")
    model_forced = model.bind_tools(
        tools=[calculate],
        # 不传 tool_choice，避免 deepseek-chat 返回 400
    )

    response = await model_forced.ainvoke("你好")

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        call_name = tool_calls[0]["name"]
        call_args = tool_calls[0].get("args", tool_calls[0].get("arguments", {}))
        print(f"模型调用: {call_name}")
        print(f"参数: {call_args}")
    else:
        print("模型未调用工具，改为应用层强制调用")
        tool_result = calculate.invoke({"expression": "2+3"})
        print(f"强制调用: calculate")
        print(f"参数: {{'expression': '2+3'}}")
        print(f"结果: {tool_result}")

        response = await model.ainvoke(
            [
                HumanMessage(
                    content="你好，请只使用 calculate 工具帮我计算：2+3等于多少"
                ),
                AIMessage(content=""),
                HumanMessage(
                    content=f"calculate 执行结果：{tool_result}，请基于该结果回答。"
                ),
            ]
        )
        print(f"最终回答: {response.content}")

    # 方式2: 允许自动决定调用哪个函数
    print("\n方式2: 自动决定调用哪个函数")
    model_auto = model.bind_tools(
        tools=[get_weather, calculate],
        tool_choice="auto",
    )

    response2 = await model_auto.ainvoke("2+3等于多少")
    print(f"输入: 2+3等于多少")

    if hasattr(response2, "additional_kwargs"):
        tool_calls2 = response2.additional_kwargs.get("tool_calls", [])
        if tool_calls2:
            print(f"自动调用: {tool_calls2[0]['function']['name']}")
        else:
            print("未触发工具调用")


# ============================================================================
# 第五部分：处理函数调用结果
# ============================================================================


async def demo_handle_function_results():
    """
    处理函数调用结果
    """
    print("\n" + "=" * 60)
    print("【5. 处理函数调用结果】")
    print("=" * 60)

    # 定义工具
    @tool
    def get_weather(location: str) -> str:
        """查询天气"""
        weather_db = {
            "北京": "晴，25°C，空气质量良好",
            "上海": "多云，28°C，有轻度污染",
            "广州": "雷阵雨，32°C",
        }
        return weather_db.get(location, f"未找到 {location} 的天气")

    # 绑定工具
    functions = [convert_to_openai_function(get_weather)]
    model_with_functions = model.bind(functions=functions)

    # 第一步：让模型决定调用函数
    print("\n第一步：模型分析问题并调用函数")

    question = "北京今天的天气怎么样？适合户外运动吗？"
    response = await model_with_functions.ainvoke(question)

    print(f"问题: {question}")

    # 提取函数调用
    tool_calls = []
    if hasattr(response, "additional_kwargs"):
        tool_calls = response.additional_kwargs.get("tool_calls", [])

    if tool_calls:
        func_name = tool_calls[0]["function"]["name"]
        func_args = eval(tool_calls[0]["function"]["arguments"])  # 解析 JSON 参数

        print(f"\n函数调用:")
        print(f"  函数名: {func_name}")
        print(f"  参数: {func_args}")

        # 第二步：执行函数
        print("\n第二步：执行函数")
        tool_result = get_weather.invoke(func_args)
        print(f"  结果: {tool_result}")

        # 第三步：将结果返回给模型生成最终答案
        print("\n第三步：将结果返回模型生成最终回答")

        # 创建包含函数调用和结果的对话
        messages = [
            HumanMessage(content=question),
            response,
            AIMessage(content=""),  # 空的 AI 消息作为函数结果的位置标记
        ]

        # 注意：这里需要将函数结果作为 ToolMessage 返回
        # 实际使用中，Agent 会自动处理这个过程
        print(f"  函数结果: {tool_result}")
        print(f"  模型将根据天气信息回答：适合户外运动...")


# ============================================================================
# 第六部分：结构化输出
# ============================================================================


async def demo_structured_output():
    """
    使用 function calling 实现结构化输出
    """
    print("\n" + "=" * 60)
    print("【6. 结构化输出】")
    print("=" * 60)

    # 定义输出结构
    class WeatherInfo(BaseModel):
        """天气信息结构"""

        city: str = Field(description="城市名称")
        temperature: str = Field(description="温度")
        weather: str = Field(description="天气状况")
        humidity: str = Field(description="湿度")
        wind: str = Field(description="风力")
        suggestion: str = Field(description="出行建议")

    # 创建输出函数
    output_functions = [
        {
            "name": "weather_extractor",
            "description": "提取天气信息并给出建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "temperature": {"type": "string", "description": "温度"},
                    "weather": {"type": "string", "description": "天气状况"},
                    "humidity": {"type": "string", "description": "湿度"},
                    "wind": {"type": "string", "description": "风力"},
                    "suggestion": {"type": "string", "description": "出行建议"},
                },
                "required": ["city", "weather", "suggestion"],
            },
        }
    ]

    # 绑定到模型
    model_structured = model.bind(
        functions=output_functions, function_call={"name": "weather_extractor"}
    )

    # 创建 Parser
    parser = JsonOutputFunctionsParser(pydantic_schema=WeatherInfo)

    # 创建 Chain
    chain = model_structured | parser

    print("\n结构化输出 Chain:")
    print("  模型 -> JSON 函数调用 -> Pydantic 模型")

    # 调用
    result = await chain.ainvoke("今天天气怎么样？适合出门吗？")

    print(f"\n结构化结果:")
    print(f"  城市: {result.city}")
    print(f"  温度: {result.temperature}")
    print(f"  天气: {result.weather}")
    print(f"  湿度: {result.humidity}")
    print(f"  风力: {result.wind}")
    print(f"  建议: {result.suggestion}")


# ============================================================================
# 第七部分：并行函数调用
# ============================================================================


async def demo_parallel_function_calls():
    """
    演示并行函数调用
    """
    print("\n" + "=" * 60)
    print("【7. 并行函数调用】")
    print("=" * 60)

    # 定义多个工具
    @tool
    def get_weather(location: str) -> str:
        """查询天气"""
        return f"{location}: 晴，25°C"

    @tool
    def get_news(category: str) -> str:
        """获取新闻"""
        return f"{category} 新闻: 1. XXX 2. YYY 3. ZZZ"

    @tool
    def get_time() -> str:
        """获取时间"""
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")

    # 绑定工具
    functions = [
        convert_to_openai_function(get_weather),
        convert_to_openai_function(get_news),
        convert_to_openai_function(get_time),
    ]

    model_with_functions = model.bind(functions=functions)

    print("已绑定 3 个工具到模型")
    print("模型可以同时调用多个函数")

    # 测试
    print("\n--- 测试：触发多个函数调用 ---")
    response = await model_with_functions.ainvoke(
        "帮我查一下北京和上海的天气，再给我看看今天的新闻，还有现在几点了？"
    )

    if hasattr(response, "additional_kwargs"):
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        print(f"触发了 {len(tool_calls)} 个函数调用:")
        for i, call in enumerate(tool_calls):
            print(
                f"  {i + 1}. {call['function']['name']}: {call['function']['arguments']}"
            )


# ============================================================================
# 第八部分：实际应用 - 完整的 Agent
# ============================================================================


async def demo_complete_agent():
    """
    完整的 Function Calling Agent 示例
    """
    print("\n" + "=" * 60)
    print("【8. 完整 Function Calling Agent】")
    print("=" * 60)

    from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
    from langchain_core import hub

    # 定义工具
    @tool
    def get_weather(location: str) -> str:
        """查询天气

        Args:
            location: 城市名称
        """
        weather_db = {
            "北京": "晴，25°C，适合户外活动",
            "上海": "多云，28°C",
            "广州": "雷阵雨，32°C，建议带伞",
        }
        return weather_db.get(location, f"未找到 {location} 的天气")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式

        Args:
            expression: 数学表达式，如 "2+3*4"
        """
        try:
            return f"{expression} = {eval(expression)}"
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def search_web(query: str) -> str:
        """搜索网络

        Args:
            query: 搜索关键词
        """
        return f"关于 '{query}' 的搜索结果：1. ... 2. ... 3. ..."

    tools = [get_weather, calculate, search_web]

    # 使用 OpenAI Functions Agent
    prompt = hub.pull("hwchase17/openai-functions")

    agent = create_openai_functions_agent(model, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    # 测试用例
    test_cases = [
        "北京今天天气怎么样？",
        "帮我算一下 100 * 99 + 1",
        "深圳适合旅游吗？说说天气情况",
    ]

    for case in test_cases:
        print(f"\n{'=' * 50}")
        print(f"问题: {case}")
        print("=" * 50)

        result = await agent_executor.ainvoke({"input": case})
        print(f"\n最终答案: {result['output']}")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """
    运行所有演示
    """
    print("=" * 60)
    print("OpenAI Function Calling 教程")
    print("=" * 60)

    # 1. 概念讲解
    # explain_function_calling()

    # # 2. Tool 转 OpenAI 格式
    # demo_convert_to_openai()

    # # 3. 绑定函数到模型
    # await demo_bind_functions()

    # 4. 强制函数调用
    await demo_force_function_call()

    # # 5. 处理函数调用结果
    # await demo_handle_function_results()

    # # 6. 结构化输出
    # await demo_structured_output()

    # 7. 并行函数调用
    # await demo_parallel_function_calls()

    # 8. 完整 Agent
    # await demo_complete_agent()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
