# -*- coding: utf-8 -*-
# 阳光逆流而上
"""
=============================================================================
LangChain Prompt Template 教程
=============================================================================

Prompt Template 是 LangChain 的核心概念之一，用于：
1. 结构化提示词，便于复用和维护
2. 动态注入变量
3. 支持多种消息类型（系统、用户、AI）
4. 支持对话历史

=============================================================================
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain_core.output_parsers import StrOutputParser


# ============================================================================
# 环境配置
# ============================================================================

# 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"

# 初始化模型（使用 DeepSeek）
model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    base_url="https://api.deepseek.com",
    temperature=0.7,
)


# ============================================================================
# 第一部分：基础 PromptTemplate
# ============================================================================


def demo_basic_prompt_template():
    """
    基础 PromptTemplate

    用于简单的文本补全任务（非对话场景）
    """
    print("\n" + "=" * 60)
    print("【1. 基础 PromptTemplate】")
    print("=" * 60)

    # 方式1: from_template - 最常用
    template = PromptTemplate.from_template("请将以下中文翻译成英文：{text}")

    # 方式2: template 属性（更灵活）
    template2 = PromptTemplate(
        template="用{style}风格回答：{question}", input_variables=["style", "question"]
    )

    # 方式3: partial - 预填充部分变量
    template3 = PromptTemplate(
        template="回答关于{topic}的问题：{question}",
        input_variables=["question"],  # topic 会通过 partial 预填充
    ).partial(topic="Python编程")

    print(f"\n方式1 模板: {template.template}")
    print(f"方式2 模板: {template2.template}")
    print(f"方式3 模板: {template3.template}")

    # 格式化模板
    formatted = template.invoke({"text": "你好世界"})
    print(f"\n格式化后: {formatted}")


async def demo_basic_prompt_with_model():
    """
    PromptTemplate 配合模型使用
    """
    print("\n" + "=" * 60)
    print("【1.2 PromptTemplate + 模型】")
    print("=" * 60)

    # 创建模板
    translator = PromptTemplate.from_template("将以下中文翻译成{source_lang}：\n{text}")

    # 创建 Chain: 模板 -> 模型 -> 输出
    chain = translator | model | StrOutputParser()

    # 调用
    result = await chain.ainvoke({"source_lang": "英文", "text": "今天天气真好！"})

    print(f"翻译结果: {result}")


# ============================================================================
# 第二部分：ChatPromptTemplate - 对话模板
# ============================================================================


def demo_chat_prompt_template():
    """
    ChatPromptTemplate 用于对话场景

    支持多种消息类型：
    1. ("system", "系统提示")
    2. ("human", "用户消息")
    3. ("ai", "AI消息")
    4. ("placeholder", 占位符)
    """
    print("\n" + "=" * 60)
    print("【2. ChatPromptTemplate - 对话模板】")
    print("=" * 60)

    # 基础对话模板
    basic_chat = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{sys_role}，专门回答关于{topic}的问题。"),
            ("human", "{user_question}"),
        ]
    )

    print("基础对话模板结构:")
    for i, msg in enumerate(basic_chat.messages):
        print(f"  消息{i + 1}: {msg}")

    # 格式化
    formatted = basic_chat.format_messages(
        sys_role="专家", topic="健康", user_question="每天应该喝多少水？"
    )

    print("\n格式化后的消息:")
    for msg in formatted:
        print(f"  {msg.type}: {msg.content}")


async def demo_chat_prompt_with_model():
    """
    ChatPromptTemplate 配合模型使用
    """
    print("\n" + "=" * 60)
    print("【2.2 ChatPromptTemplate + 模型】")
    print("=" * 60)

    # 创建对话模板
    qa_chat = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个友好的助手。请根据上下文回答问题。

上下文：
{context}

要求：
1. 如果上下文中有相关信息，必须基于上下文回答
2. 如果上下文没有相关信息，诚实说明不知道
3. 回答要简洁明了""",
            ),
            ("human", "{question}"),
        ]
    )

    # 创建 Chain
    chain = qa_chat | model | StrOutputParser()

    # 调用
    result = await chain.ainvoke(
        {
            "context": "Python 是一种高级编程语言，由 Guido van Rossum 于1991年创建。",
            "question": "Python 是谁创建的？",
        }
    )

    print(f"问答结果: {result}")


# ============================================================================
# 第三部分：MessagesPlaceholder - 动态消息列表
# ============================================================================


def demo_messages_placeholder():
    """
    MessagesPlaceholder 用于动态插入消息列表

    典型场景：
    1. 对话历史
    2. 多轮对话
    3. Agent 的中间步骤
    """
    print("\n" + "=" * 60)
    print("【3. MessagesPlaceholder - 消息占位符】")
    print("=" * 60)

    # 创建支持对话历史的模板
    chat_with_history = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{sys_role}。"),
            MessagesPlaceholder(variable_name="chat_history"),  # 动态插入历史消息
            ("human", "{current_input}"),
        ]
    )

    print("模板结构:")
    for i, msg in enumerate(chat_with_history.messages):
        if hasattr(msg, "variable_name"):
            print(
                f"  消息{i + 1}: MessagesPlaceholder(variable_name='{msg.variable_name}')"
            )
        else:
            print(f"  消息{i + 1}: {msg}")

    # 模拟对话历史
    history = [
        HumanMessage(content="我想学习编程"),
        AIMessage(content="很好！请问你想学习什么编程语言？"),
        HumanMessage(content="Python"),
    ]

    # 格式化
    formatted = chat_with_history.format_messages(
        sys_role="编程导师", chat_history=history, current_input="Python 适合初学者吗？"
    )

    print("\n格式化后的消息:")
    for msg in formatted:
        role = msg.type.upper()
        print(f"  [{role}]: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")


async def demo_conversation_chain():
    """
    完整的多轮对话 Chain
    """
    print("\n" + "=" * 60)
    print("【3.2 多轮对话 Chain】")
    print("=" * 60)

    # 创建对话模板
    conversation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的对话助手，名字叫小智。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # 创建 Chain
    conversation_chain = conversation_prompt | model | StrOutputParser()

    # 模拟多轮对话
    history = []

    questions = [
        "我叫小明，请问你叫什么名字？",
        "小明是谁？",
    ]

    for q in questions:
        # 添加用户消息到历史
        history.append(HumanMessage(content=q))

        # 调用 Chain
        response = await conversation_chain.ainvoke({"input": q, "history": history})

        print(f"\n用户: {q}")
        print(f"助手: {response}")

        # 添加 AI 回复到历史
        history.append(AIMessage(content=response))


# ============================================================================
# 第四部分：PipelinePrompt - 组合式提示词
# ============================================================================


def demo_pipeline_prompt():
    """
    PipelinePrompt 用于组合多个提示词模板

    这个版本在当前 langchain-core 中不可用，改为手动拼接示例。
    """
    print("\n" + "=" * 60)
    print("【4. 管道提示词（手动拼接版）】")
    print("=" * 60)

    # 定义各个部分
    introduction = PromptTemplate.from_template("你是{role}。")
    examples = PromptTemplate.from_template("""以下是一些示例：
{examples}""")
    transition = PromptTemplate.from_template("现在请回答用户的问题。")
    question = PromptTemplate.from_template("{question}")

    # 手动按顺序组合各段模板
    formatted = "\n".join(
        [
            introduction.format(role="数学老师"),
            examples.format(examples="1+1=2\n2+2=4"),
            transition.format(),
            question.format(question="3+3=?"),
        ]
    )

    print("Pipeline 结构:")
    for name, prompt in [
        ("introduction", introduction),
        ("examples", examples),
        ("transition", transition),
        ("question", question),
    ]:
        print(f"  {name}: {prompt.template[:50]}...")

    print(f"\n最终提示词:\n{formatted}")


# ============================================================================
# 第五部分：少样本提示词（Few-shot）
# ============================================================================


def demo_few_shot_prompt():
    """
    Few-shot Prompt 用于提供示例帮助模型理解任务
    """
    print("\n" + "=" * 60)
    print("【5. Few-shot Prompt - 少样本提示词】")
    print("=" * 60)

    # 定义示例
    examples = [
        {"input": "今天天气真好", "output": "积极"},
        {"input": "我考试没考好", "output": "消极"},
        {"input": "电影一般般", "output": "中性"},
    ]

    # 创建示例模板
    example_prompt = PromptTemplate.from_template("输入: {input}\n输出: {output}")

    # 创建 Few-shot 模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="判断以下句子的情感（积极/消极/中性）：",
        suffix="\n输入: {input}\n输出:",
        input_variables=["input"],
    )

    print("Few-shot 模板:")
    print(f"前缀: {few_shot_prompt.prefix}")
    print(f"后缀: {few_shot_prompt.suffix}")
    print(f"示例数量: {len(few_shot_prompt.examples)}")

    # 格式化
    formatted = few_shot_prompt.format(input="我很开心！")
    print(f"\n格式化结果:\n{formatted}")


# ============================================================================
# 第六部分：提示词模板进阶技巧
# ============================================================================


def demo_advanced_techniques():
    """
    进阶技巧：
    1. partial - 预填充变量
    2. compact - 压缩长提示词
    3. 自定义验证器
    """
    print("\n" + "=" * 60)
    print("【6. 进阶技巧】")
    print("=" * 60)

    # 1. partial - 预填充
    base_template = PromptTemplate.from_template("{operation} {subject} 的 {attribute}")

    # 预填充 operation
    operation_partial = base_template.partial(operation="分析")

    # 现在只需要填充 subject 和 attribute
    formatted = operation_partial.format(subject="Python", attribute="优点")
    print(f"1. partial 预填充: {formatted}")

    # 2. 默认值
    template_with_default = PromptTemplate.from_template(
        "回答关于 {topic} 的问题。\n{extra:默认值：这是一个额外信息}"
    )

    # 带 extra
    print(
        f"2a. 有 extra: {template_with_default.format(topic='AI', extra='AI 是人工智能')}"
    )

    # 不带 extra（使用默认值）
    print(f"2b. 无 extra: {template_with_default.format(topic='AI')}")


# ============================================================================
# 第七部分：实践练习
# ============================================================================


async def practice_exercise():
    """
    练习：构建一个多轮问答系统
    """
    print("\n" + "=" * 60)
    print("【练习】多轮问答系统")
    print("=" * 60)

    # 创建问答模板
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个知识渊博的老师。

特点：
- 回答简洁明了
- 善于用例子解释概念
- 如果不知道就说不知道

对话历史：
{chat_history}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # 创建 Chain
    qa_chain = qa_prompt | model | StrOutputParser()

    # 初始化对话历史
    chat_history = []

    # 多轮对话
    questions = [
        "什么是人工智能？",
        "它和机器学习有什么关系？",
        "深度学习是什么？",
    ]

    for q in questions:
        print(f"\n用户: {q}")

        # 构建输入
        inputs = {
            "question": q,
            "chat_history": "\n".join(
                [
                    f"用户: {m.content}"
                    if isinstance(m, HumanMessage)
                    else f"助手: {m.content}"
                    for m in chat_history
                ]
            ),
            "history": chat_history,
        }

        # 调用
        response = await qa_chain.ainvoke(inputs)
        print(f"助手: {response}")

        # 更新历史
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=response))


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """
    运行所有演示
    """
    print("=" * 60)
    print("LangChain Prompt Template 教程")
    print("=" * 60)

    # 1. 基础 PromptTemplate
    demo_basic_prompt_template()
    await demo_basic_prompt_with_model()

    # 2. ChatPromptTemplate
    demo_chat_prompt_template()
    await demo_chat_prompt_with_model()

    # 3. MessagesPlaceholder
    demo_messages_placeholder()
    await demo_conversation_chain()

    # 4. PipelinePrompt
    demo_pipeline_prompt()

    # 5. Few-shot
    demo_few_shot_prompt()

    # 6. 进阶技巧
    demo_advanced_techniques()

    # 7. 实践练习
    await practice_exercise()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
