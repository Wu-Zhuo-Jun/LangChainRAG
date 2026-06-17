# -*- coding: utf-8 -*-
"""
=============================================================================
LangChain Memory 教程
=============================================================================

Memory（记忆）是 Agent 保持对话上下文的关键组件。

Memory 类型：
1. ConversationBufferMemory - 简单缓冲记忆
2. ConversationBufferWindowMemory - 窗口记忆
3. ConversationTokenBufferMemory - Token 限制记忆
4. ConversationSummaryMemory - 摘要记忆
5. VectorStoreRetrieverMemory - 向量存储记忆

=============================================================================
"""

import os
from typing import Any
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.factory import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool


# ============================================================================
# 环境配置
# ============================================================================

os.environ["OPENAI_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    base_url="https://api.deepseek.com",
    temperature=0.7,
)


# ============================================================================
# 第一部分：Memory 基础概念
# ============================================================================


def demo_memory_concepts():
    """
    Memory 的核心概念

    Memory 主要功能：
    1. 保存对话历史
    2. 在每次调用时传递给 Agent
    3. 支持不同类型的存储和检索策略
    """
    print("\n" + "=" * 60)
    print("【1. Memory 核心概念】")
    print("=" * 60)

    print("""
Memory 在 Agent 中的位置：

┌─────────────────────────────────────────────────────────────┐
│                        Agent 系统                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   用户输入 ──┬───────────────────────────────────────────▶   │
│              │                                                │
│              │                                                │
│              ▼                                                │
│   ┌──────────────────┐                                      │
│   │      Memory      │ ◀──── 保存对话历史                    │
│   │  (对话记忆)       │                                      │
│   └────────┬─────────┘                                      │
│            │                                                │
│            │ 读取相关历史                                     │
│            ▼                                                │
│   ┌──────────────────┐                                      │
│   │   Prompt + History │ ──────────────────▶ LLM             │
│   │  (带历史的提示)    │                                      │
│   └──────────────────┘                                      │
│                                                              │
│   输出 + 保存 ─────────────────────────────────────────▶    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

""")

    input("\n按 Enter 继续...")


# ============================================================================
# 第二部分：ConversationBufferMemory
# ============================================================================


def demo_buffer_memory():
    """
    ConversationBufferMemory - 最简单的记忆类型

    特点：
    - 保存所有对话历史
    - 不限制长度
    - 简单直接
    """
    print("\n" + "=" * 60)
    print("【2. ConversationBufferMemory - 缓冲记忆】")
    print("=" * 60)

    # 创建记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 在 Prompt 中引用的键名
        return_messages=True,  # 返回消息对象列表，而不是字符串
        output_key="output",  # 从输出中提取什么作为 AI 回复
        input_key="input",  # 从输入中提取什么作为用户输入
    )

    print("创建 ConversationBufferMemory")
    print(f"memory_key: {memory.memory_key}")
    print(f"return_messages: {memory.return_messages}")

    # 保存对话
    memory.save_context(
        {"input": "我叫小明"}, {"output": "你好小明，我叫小智，很高兴认识你！"}
    )

    memory.save_context(
        {"input": "我喜欢编程"},
        {"output": "太棒了！编程是一项很有趣的技能。你想学习什么语言？"},
    )

    # 加载记忆
    print("\n--- 保存的对话历史 ---")
    chat_history = memory.load_memory_variables({})

    print(f"\nchat_history 类型: {type(chat_history['chat_history'])}")
    print(f"消息数量: {len(chat_history['chat_history'])}")

    for i, msg in enumerate(chat_history["chat_history"]):
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i + 1}. [{role}] {msg.content}")


async def demo_buffer_memory_with_llm():
    """
    使用 BufferMemory 进行多轮对话
    """
    print("\n" + "=" * 60)
    print("【2.2 BufferMemory + LLM 多轮对话】")
    print("=" * 60)

    # 创建记忆
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的助手，名字叫小智。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # 创建 Chain
    chain = prompt | model | StrOutputParser()

    # 多轮对话
    questions = [
        "我叫小明，请问你叫什么名字？",
        "你知道小明是谁吗？",
        "我们第一次见面时我做了什么自我介绍？",
    ]

    for q in questions:
        print(f"\n用户: {q}")

        # 获取历史
        chat_history = memory.load_memory_variables({})["chat_history"]

        # 调用 Chain
        response = await chain.ainvoke({"input": q, "chat_history": chat_history})

        print(f"助手: {response}")

        # 保存到记忆
        memory.save_context({"input": q}, {"output": response})


# ============================================================================
# 第三部分：ConversationBufferWindowMemory
# ============================================================================


def demo_window_memory():
    """
    ConversationBufferWindowMemory - 窗口记忆

    特点：
    - 只保留最近 K 轮对话
    - 防止记忆无限增长
    """
    print("\n" + "=" * 60)
    print("【3. ConversationBufferWindowMemory - 窗口记忆】")
    print("=" * 60)

    # 创建窗口记忆，只保留最近 3 轮
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=3,  # 只保留最近 3 轮对话
    )

    # 添加多轮对话
    conversations = [
        ("我叫张三", "你好张三！"),
        ("我喜欢打篮球", "篮球是个好运动！"),
        ("我住在上海", "上海是个国际化大都市。"),
        ("今天天气很好", "是啊，阳光明媚。"),
        ("你是哪里开发的", "我是由 AI 技术驱动的。"),
    ]

    print("添加 5 轮对话，窗口大小 k=3：\n")

    for user_input, ai_output in conversations:
        memory.save_context({"input": user_input}, {"output": ai_output})
        chat_history = memory.load_memory_variables({})["chat_history"]
        print(f"添加: '{user_input}'")
        print(f"  当前记忆: {len(chat_history)} 条消息")

    print("\n--- 最终记忆（只保留最近 3 轮）---")
    history = memory.load_memory_variables({})["chat_history"]

    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content}")


# ============================================================================
# 第四部分：ConversationTokenBufferMemory
# ============================================================================


def demo_token_memory():
    """
    ConversationTokenBufferMemory - Token 限制记忆

    特点：
    - 根据 Token 数量限制记忆长度
    - 更精确地控制上下文长度
    """
    print("\n" + "=" * 60)
    print("【4. ConversationTokenBufferMemory - Token 限制】")
    print("=" * 60)

    # 创建 Token 限制记忆（限制为约 500 tokens）
    memory = ConversationTokenBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=500,  # 限制 500 tokens
        llm=model,  # 需要 LLM 来计算 token 数
    )

    print(f"Token 限制: {memory.max_token_limit}")

    # 添加对话
    conversations = [
        (
            "给我讲一个关于人工智能的故事，要求详细生动",
            "很久以前，有一个小村庄，那里的人们过着平静的生活。一天，一个名叫李明的年轻科学家带来了一个人工智能助手。这个助手能够回答各种问题，帮助村民解决难题。渐渐地，AI成为了村庄不可或缺的一部分。",
        ),
        (
            "这个故事后续如何发展？",
            "随着时间推移，人工智能越来越先进。它不仅能回答问题，还能帮助村民进行医疗诊断、教育学习。村庄变得更加繁荣，人们与AI和谐共处。",
        ),
        (
            "故事的寓意是什么？",
            "这个故事告诉我们，科技应该服务于人类，带来美好和便利。",
        ),
    ]

    print("\n添加 3 轮较长对话，Token 限制 500：\n")

    for user_input, ai_output in conversations:
        memory.save_context({"input": user_input}, {"output": ai_output})
        chat_history = memory.load_memory_variables({})["chat_history"]
        total_chars = sum(len(msg.content) for msg in chat_history)
        print(f"添加后: {len(chat_history)} 条消息，约 {total_chars} 字符")

    print("\n--- 超出 Token 限制后的记忆 ---")
    history = memory.load_memory_variables({})["chat_history"]

    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{role}] {preview}")


# ============================================================================
# 第五部分：ConversationSummaryMemory
# ============================================================================


async def demo_summary_memory():
    """
    ConversationSummaryMemory - 摘要记忆

    特点：
    - 不保存完整对话，而是生成摘要
    - 节省 Token，适合长对话
    """
    print("\n" + "=" * 60)
    print("【5. ConversationSummaryMemory - 摘要记忆】")
    print("=" * 60)

    # 创建摘要记忆
    memory = ConversationSummaryMemory(
        memory_key="chat_history",
        return_messages=True,
        llm=model,  # 需要 LLM 来生成摘要
    )

    print("添加多轮对话：\n")

    # 长对话
    conversations = [
        (
            "我叫王芳，是一名软件工程师，在北京工作",
            "你好王芳！软件工程师是个很有前景的职业。北京作为首都，有很多科技公司。",
        ),
        (
            "是的，我目前在一家 AI 公司做后端开发",
            "AI 公司的后端开发很有挑战性！你在做什么具体的工作？",
        ),
        (
            "我主要负责开发机器学习模型的部署系统",
            "模型部署是 AI 落地的重要环节，非常关键！",
        ),
        ("没错，我们团队最近在优化推理性能", "性能优化是个技术活，有什么进展吗？"),
    ]

    for user_input, ai_output in conversations:
        print(f"用户: {user_input[:30]}...")
        memory.save_context({"input": user_input}, {"output": ai_output})

        # 查看当前摘要
        summary = memory.load_memory_variables({})
        summary_data = summary.get("chat_history", [])
        if summary_data:
            if isinstance(summary_data, str):
                # return_messages=False 时返回字符串
                print(f"  摘要: {summary_data[:80]}...")
            else:
                # return_messages=True 时返回消息列表
                for msg in summary_data:
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                        if len(content) > 80:
                            print(f"  摘要: {content[:80]}...")
                        else:
                            print(f"  摘要: {content}")

    print("\n--- 最终摘要 ---")
    final_summary = memory.load_memory_variables({})
    final_data = final_summary.get("chat_history")
    if final_data:
        if isinstance(final_data, str):
            print(final_data)
        else:
            for msg in final_data:
                if hasattr(msg, "content") and msg.content:
                    print(msg.content)


# ============================================================================
# 第六部分：VectorStoreRetrieverMemory - 向量记忆
# ============================================================================


def demo_vector_memory():
    """
    VectorStoreRetrieverMemory - 向量检索记忆

    特点：
    - 使用向量数据库存储记忆
    - 根据语义相似度检索相关记忆
    - 最适合大规模长期记忆
    """
    print("\n" + "=" * 60)
    print("【6. VectorStoreRetrieverMemory - 向量记忆】")
    print("=" * 60)

    # 创建向量存储
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com"
    )

    # 使用简单的内存向量存储
    vectorstore = FAISS.from_texts(
        ["初始空的记忆"], embeddings, metadatas=[{"source": "init"}]
    )

    # 创建向量记忆
    memory = VectorStoreRetrieverMemory(
        vectorstore=vectorstore,
        memory_key="chat_history",
        return_messages=True,
        search_kwargs={"k": 3},  # 检索最近 3 条相关记忆
    )

    print("向量记忆系统已创建")
    print(f"搜索参数: k=3（检索3条相关记忆）")

    # 保存一些记忆
    memories = [
        ("用户喜欢蓝色，不喜欢红色", "了解用户颜色偏好：蓝色"),
        ("用户的生日是 6 月 1 日", "记住用户生日：6月1日"),
        ("用户在找一份 Python 开发的工作", "了解用户求职意向：Python开发"),
        ("用户住在深圳，有一只猫", "用户住在深圳，养猫"),
        ("用户最喜欢的食物是火锅", "记住用户喜好：火锅"),
    ]

    for fact, memory_text in memories:
        memory.save_context({"input": fact}, {"output": memory_text})

    # 检索相关记忆
    test_queries = [
        "用户有什么饮食偏好？",
        "用户的工作相关？",
        "用户的个人信息？",
    ]

    print("\n--- 检索测试 ---")
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = memory.load_memory_variables({"input": query})
        print(f"相关记忆:")
        for msg in results.get("chat_history", []):
            if isinstance(msg, HumanMessage):
                print(f"  事实: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"  记忆: {msg.content}")


# ============================================================================
# 第七部分：带 Memory 的 Agent
# ============================================================================


async def demo_agent_with_memory():
    """
    完整的带 Memory 的 Agent 示例
    """
    print("\n" + "=" * 60)
    print("【7. 带 Memory 的 Agent】")
    print("=" * 60)

    # 定义简单工具
    @tool
    def get_weather(location: str) -> str:
        """查询天气"""
        return f"{location}今天晴天，25°C"

    @tool
    def get_time() -> str:
        """获取当前时间"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools = [get_weather, get_time]

    # 创建记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # 创建自定义 Prompt
    system_prompt = """你是一个智能助手，可以回答问题和调用工具。

记住之前的对话内容。
"""

    # 创建 Agent (使用新版 API)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    # 多轮对话
    questions = [
        "我叫张三，住在深圳",
        "今天天气怎么样？",
        "你还记得我叫什么名字吗？",
        "我现在在哪里？",
    ]

    for q in questions:
        print(f"\n{'=' * 50}")
        print(f"用户: {q}")
        print("=" * 50)

        # 从记忆中加载历史消息
        history_messages = memory.load_memory_variables({}).get("chat_history", [])

        # 构建带历史的消息列表
        from langchain_core.messages import HumanMessage

        messages = list(history_messages) + [HumanMessage(content=q)]

        # 使用新版 API 执行 Agent
        config: RunnableConfig = {"recursion_limit": 10}
        result = await agent.ainvoke({"messages": messages}, config)

        # 提取回复并保存到记忆
        if isinstance(result, dict) and "messages" in result:
            response = result["messages"][-1].content
            print(f"助手: {response}")
            # 保存对话到记忆（只保存新的对话）
            memory.save_context({"input": q}, {"output": response})
        else:
            # 处理其他返回格式
            print(f"助手: {result}")

    # 查看记忆内容
    print("\n--- 最终对话记忆 ---")
    history = memory.load_memory_variables({})["chat_history"]
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content[:60]}{'...' if len(msg.content) > 60 else ''}")


# ============================================================================
# 第八部分：自定义 Memory
# ============================================================================


def demo_custom_memory():
    """
    自定义 Memory 的方法
    """
    print("\n" + "=" * 60)
    print("【8. 自定义 Memory】")
    print("=" * 60)

    from langchain_classic.memory.chat_memory import BaseChatMemory
    from langchain_core.messages import get_buffer_string

    class CustomMemory(BaseChatMemory):
        """
        自定义记忆类

        可以实现：
        - 从数据库加载/保存记忆
        - 从文件加载/保存记忆
        - 实现特定的数据处理逻辑
        """

        def load_memory_variables(self, inputs: dict) -> dict:
            """
            加载记忆变量

            返回一个包含 'chat_history' 键的字典
            """
            # 这里可以从任何地方加载记忆
            # 例如：数据库、文件、API
            return {"chat_history": self.chat_memory.messages}

        @property
        def memory_variables(self) -> list:
            """
            定义这个 Memory 提供的变量
            """
            return ["chat_history"]

        def save_context(self, inputs: dict, outputs: dict) -> None:
            """
            保存对话上下文
            """
            # 这里可以添加自定义保存逻辑
            # 例如：保存到数据库
            pass

        def clear(self) -> None:
            """
            清除记忆
            """
            self.chat_memory.clear()

    print("自定义 Memory 类已定义")
    print("需要实现的方法:")
    print("  1. load_memory_variables() - 加载记忆")
    print("  2. memory_variables - 返回变量名列表")
    print("  3. save_context() - 保存上下文")
    print("  4. clear() - 清除记忆")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """
    运行所有演示
    """
    print("=" * 60)
    print("LangChain Memory 教程")
    print("=" * 60)

    # 1. 核心概念
    # demo_memory_concepts()

    # 2. BufferMemory
    # demo_buffer_memory()
    # await demo_buffer_memory_with_llm()

    # 3. WindowMemory
    # demo_window_memory()

    # 4. TokenMemory
    # demo_token_memory()

    # 5. SummaryMemory
    # await demo_summary_memory()

    # 6. VectorMemory
    # demo_vector_memory()

    # 7. Agent + Memory
    await demo_agent_with_memory()

    # 8. 自定义 Memory
    # demo_custom_memory()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


# Python 程序入口点：
# - __name__ == "__main__" 确保只在直接运行此文件时执行（被 import 时不运行）
# - asyncio.run() 用于运行 async 协程函数
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
