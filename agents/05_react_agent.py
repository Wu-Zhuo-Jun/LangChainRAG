# -*- coding: utf-8 -*-
"""
=============================================================================
LangChain ReAct Agent 教程
=============================================================================

ReAct = Reasoning + Acting

ReAct 是一种让 Agent 能够进行多步骤推理和行动的框架。

核心循环：
1. Thought (思考) - 分析当前情况，决定下一步
2. Action (行动) - 执行工具
3. Observation (观察) - 获取结果
4. 重复直到完成任务

=============================================================================
"""

# ============================================================================
# ⚠️ 警告：这个文件使用的是旧版 LangChain API，已不兼容 LangChain 1.3.x
# 请使用新版教程: agents/05_react_agent_v2.py
# ============================================================================

import os
from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool, BaseTool

# 旧版 API 已弃用，尝试导入（可能失败）
try:
    from langchain.agents import (
        AgentExecutor,
        create_react_agent,
        create_openai_functions_agent,
    )
except ImportError:
    AgentExecutor = None
    create_react_agent = None
    create_openai_functions_agent = None
    print("⚠️ 警告: 旧版 Agent API 已弃用，请使用新版 agents/05_react_agent_v2.py")

from langchain import hub  # noqa: F401
from langchain_core.output_parsers import StrOutputParser  # noqa: F401
from pydantic import BaseModel, Field  # noqa: F401


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
    print("\n" + "="*60)
    print("【1. ReAct 原理详解】")
    print("="*60)
    
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
Action: 根据以上信息给出建议
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
                "wind": "2-3级"
            },
            "上海": {
                "weather": "多云",
                "temperature": "28°C",
                "air_quality": "轻度污染",
                "humidity": "65%",
                "wind": "3-4级"
            },
            "广州": {
                "weather": "雷阵雨",
                "temperature": "32°C",
                "air_quality": "优",
                "humidity": "80%",
                "wind": "4-5级"
            },
            "深圳": {
                "weather": "晴",
                "temperature": "30°C",
                "air_quality": "良",
                "humidity": "55%",
                "wind": "2级"
            },
        }
        
        if location in weather_db:
            info = weather_db[location]
            return (f"{location}天气：{info['weather']}，"
                    f"温度：{info['temperature']}，"
                    f"空气质量：{info['air_quality']}，"
                    f"湿度：{info['humidity']}，"
                    f"风力：{info['wind']}")
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
            # 注意：实际应用中应使用安全的计算方法
            # 这里使用 eval 仅用于演示
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
        # 模拟搜索结果
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
            return "\n".join([f"• {news}" for news in results])
        else:
            return f"暂无关于'{keyword}'的最新新闻"
    
    @tool
    def get_date_info() -> str:
        """获取当前日期和时间信息
        
        Returns:
            当前日期、星期、农历信息
        """
        from datetime import datetime
        
        now = datetime.now()
        weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
        
        return (f"今天是 {now.strftime('%Y年%m月%d日')}，"
                f"星期{weekday_names[now.weekday()]}，"
                f"时间 {now.strftime('%H:%M:%S')}")
    
    return [get_weather, calculate, search_news, get_date_info]


# ============================================================================
# 第三部分：创建 ReAct Agent
# ============================================================================

async def demo_react_agent():
    """
    演示 ReAct Agent 的创建和使用
    """
    print("\n" + "="*60)
    print("【2. 创建 ReAct Agent】")
    print("="*60)
    
    # 获取工具
    tools = create_tools()
    
    # 方式1: 使用 LangChain Hub 中的 ReAct Prompt
    print("\n--- 方式1: 使用预置的 ReAct Prompt ---")
    react_prompt = hub.pull("hwchase17/react")
    
    print(f"Prompt 模板: {react_prompt}")
    
    # 创建 Agent
    agent = create_react_agent(model, tools, react_prompt)
    
    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 打印完整思考过程
        max_iterations=10,  # 最大迭代次数，防止无限循环
        handle_parsing_errors=True,  # 处理解析错误
    )
    
    print("\n" + "="*60)
    print("【3. ReAct Agent 执行示例】")
    print("="*60)
    
    # 示例1: 天气查询
    print("\n--- 示例1: 天气查询 ---")
    question = "北京今天的天气怎么样？适合出门吗？"
    print(f"问题: {question}")
    
    result = await agent_executor.ainvoke({"input": question})
    print(f"\n最终答案: {result['output']}")
    
    # 示例2: 需要多步推理
    print("\n--- 示例2: 多步推理 ---")
    question2 = "北京和上海哪个城市更适合户外跑步？"
    print(f"问题: {question2}")
    
    result2 = await agent_executor.ainvoke({"input": question2})
    print(f"\n最终答案: {result2['output']}")


async def demo_openai_functions_agent():
    """
    演示 OpenAI Functions Agent
    
    这种 Agent 使用 OpenAI 的函数调用格式
    更加结构化和可靠
    """
    print("\n" + "="*60)
    print("【4. OpenAI Functions Agent】")
    print("="*60)
    
    tools = create_tools()
    
    # 使用 OpenAI Functions Agent
    agent = create_openai_functions_agent(model, tools, prompt=hub.pull("hwchase17/openai-functions"))
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
    )
    
    # 测试
    questions = [
        "今天是什么日期？",
        "帮我算一下 (100 + 200) * 2 等于多少？",
        "最近有什么 AI 相关的新闻？",
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        result = await agent_executor.ainvoke({"input": q})
        print(f"答案: {result['output']}")


# ============================================================================
# 第四部分：自定义 ReAct Prompt
# ============================================================================

def demo_custom_react_prompt():
    """
    自定义 ReAct Prompt
    """
    print("\n" + "="*60)
    print("【5. 自定义 ReAct Prompt】")
    print("="*60)
    
    custom_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个智能助手，名为小智。你能够使用工具来回答问题。

可用工具：
{tools}

使用规则：
1. 每个工具都有名称、描述和参数定义
2. 在调用工具前，仔细思考需要什么参数
3. 观察工具返回的结果，基于结果决定下一步行动
4. 最终给出一个完整、有帮助的回答

回复格式：
Question: 用户的问题
Thought: 你的思考过程
Action: 工具名称
Action Input: 工具参数（JSON格式）
Observation: 工具返回的结果
... (这个思考-行动-观察循环可以重复多次)
Thought: 我现在知道答案了
Final Answer: 最终答案
"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    print("自定义 Prompt 结构:")
    for msg in custom_prompt.messages:
        if hasattr(msg, 'variable_name'):
            print(f"  {msg.type}: {msg.variable_name}")
        else:
            print(f"  {msg.type}: {msg.content[:50]}...")
    
    return custom_prompt


# ============================================================================
# 第五部分：Agent 的流式输出
# ============================================================================

async def demo_streaming():
    """
    演示 Agent 的流式输出
    """
    print("\n" + "="*60)
    print("【6. Agent 流式输出】")
    print("="*60)
    
    tools = create_tools()
    
    # 使用简单的 ReAct Agent
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(model, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
    )
    
    print("流式输出演示:")
    print("-" * 40)
    
    # 流式获取输出
    async for chunk in agent_executor.astream({"input": "什么是 AI？"}):
        # 打印每个流式块
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"\n[思考] {action.log}", end="", flush=True)
        if "steps" in chunk:
            for step in chunk["steps"]:
                print(f"\n[行动] {step.observation}", end="", flush=True)
        if "output" in chunk:
            print(f"\n\n[最终答案] {chunk['output']}", end="", flush=True)
    
    print("\n" + "-" * 40)


# ============================================================================
# 第六部分：带记忆的 Agent
# ============================================================================

async def demo_agent_with_memory():
    """
    演示带记忆的 Agent
    """
    print("\n" + "="*60)
    print("【7. 带记忆的 Agent】")
    print("="*60)
    
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    tools = create_tools()
    
    # 自定义 Prompt，支持对话历史
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个智能助手。你有多种工具可以使用。

工具列表：
{tools}

记住之前的对话内容，综合考虑给出回答。
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建 Agent
    agent = create_react_agent(model, tools, prompt)
    
    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
    )
    
    # 多轮对话
    questions = [
        "我叫小明，请问你叫什么名字？",
        "小明是谁？",
        "你能帮我查询北京的天气吗？",
    ]
    
    chat_history = []
    
    for q in questions:
        print(f"\n{'='*40}")
        print(f"用户: {q}")
        print("=" * 40)
        
        # 添加用户消息
        chat_history.append(HumanMessage(content=q))
        
        # 调用 Agent
        result = await agent_executor.ainvoke({
            "input": q,
            "chat_history": chat_history,
            "tools": "\n".join([f"- {t.name}: {t.description}" for t in tools])
        })
        
        print(f"\n助手: {result['output']}")
        
        # 添加助手回复到历史
        chat_history.append(AIMessage(content=result["output"]))


# ============================================================================
# 第七部分：错误处理和调试
# ============================================================================

async def demo_error_handling():
    """
    演示 Agent 的错误处理
    """
    print("\n" + "="*60)
    print("【8. 错误处理和调试】")
    print("="*60)
    
    tools = create_tools()
    
    # 创建 Agent
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(model, tools, react_prompt)
    
    # 设置更长的最大迭代次数和错误处理
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,  # 限制迭代次数
        max_execution_time=30,  # 限制执行时间
        handle_parsing_errors="handle_parsing_errors",  # 自动处理解析错误
        early_stopping_method="force",  # 强制停止方法
    )
    
    # 测试一些边界情况
    test_cases = [
        "你好，你是谁？",  # 无需工具的问题
        "xyzabc123 是什么？",  # 无效查询
    ]
    
    for case in test_cases:
        print(f"\n测试: {case}")
        try:
            result = await agent_executor.ainvoke({"input": case})
            print(f"结果: {result['output']}")
        except Exception as e:
            print(f"错误: {type(e).__name__}: {e}")


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """
    运行所有演示
    """
    print("="*60)
    print("LangChain ReAct Agent 教程")
    print("="*60)
    
    # 1. 原理讲解
    explain_react原理()
    
    # 2. ReAct Agent
    await demo_react_agent()
    
    # 3. OpenAI Functions Agent
    await demo_openai_functions_agent()
    
    # 4. 自定义 Prompt
    demo_custom_react_prompt()
    
    # 5. 流式输出
    await demo_streaming()
    
    # 6. 带记忆的 Agent
    await demo_agent_with_memory()
    
    # 7. 错误处理
    await demo_error_handling()
    
    print("\n" + "="*60)
    print("教程完成！")
    print("="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
