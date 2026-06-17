# -*- coding: utf-8 -*-
"""
=============================================================================
LangChain Chain 教程
=============================================================================

Chain（链）是 LangChain 的核心概念，用于将多个组件组合成流水线。

本示例涵盖：
1. LLMChain / LCEL 基础链
2. SequentialChain 顺序执行
3. TransformationChain 数据转换
4. Router 风格的按条件分发

运行前请先设置环境变量：
- OPENAI_API_KEY 或 DEEPSEEK_API_KEY
- 如需自定义接口地址，可设置 DEEPSEEK_BASE_URL

说明：
- 当前环境中的 `langchain` 不包含旧版 `langchain.chains` 模块，因此本脚本统一使用
  LangChain Expression Language（LCEL）与 Runnable 方式实现“链式”示例。
- 这样可以避免版本差异导致的导入错误，并提升脚本在新版本 LangChain 中的兼容性。

=============================================================================
"""

import asyncio
import os
from typing import Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel


# ============================================================================
# 环境配置
os.environ["DEEPSEEK_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
# ============================================================================


def create_model():
    """创建聊天模型。"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未检测到 API Key。请先设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
    )


model = create_model()


# ============================================================================
# 第一部分：LLMChain - 基础链
# ============================================================================


def demo_llm_chain():
    """
    LLMChain 是最基础的链

    结构：PromptTemplate -> LLM -> OutputParser
    使用 | 运算符连接各组件
    """
    print("\n" + "=" * 60)
    print("【1. LLMChain - 基础链】")
    print("=" * 60)

    template = PromptTemplate.from_template("用{style}风格介绍{topic}")
    chain = template | model | StrOutputParser()

    print("Chain 结构:")
    print(f"  输入变量: {template.input_variables}")
    print("  输出类型: str")
    print(f"  第一步: {template}")
    print(f"  第二步: {model}")
    print(f"  第三步: {StrOutputParser()}")

    return chain


async def demo_llm_chain_invoke():
    """
    LLMChain 调用示例
    """
    print("\n" + "=" * 60)
    print("【1.2 LLMChain 调用】")
    print("=" * 60)

    translator = (
        PromptTemplate.from_template("将以下中文翻译成{lang}：\n{text}")
        | model
        | StrOutputParser()
    )

    result = await translator.ainvoke({"lang": "英文", "text": "今天天气真好！"})
    print(f"翻译结果: {result}")

    print("\n批量翻译:")
    texts = ["你好", "再见", "谢谢"]
    for text in texts:
        translated = await translator.ainvoke({"lang": "英文", "text": text})
        print(f"  {text} -> {translated}")


# ============================================================================
# 第二部分：SequentialChain - 顺序链
# ============================================================================


async def demo_sequential_chain():
    """
    使用 LCEL 模拟顺序执行多个步骤。

    在新版 LangChain 中，这种写法比旧版 SequentialChain 更稳定，
    同时仍然能清晰展示“上一步输出作为下一步输入”的核心思想。
    """
    print("\n" + "=" * 60)
    print("【2. Sequential 风格顺序链】")
    print("=" * 60)

    name_chain = (
        PromptTemplate.from_template("为一款{product_type}想5个有创意的名称")
        | model
        | StrOutputParser()
    )

    select_chain = (
        PromptTemplate.from_template(
            "从以下名称中选择最佳的一个，并说明理由：\n{product_names}"
        )
        | model
        | StrOutputParser()
    )

    marketing_chain = (
        PromptTemplate.from_template(
            "根据以下最佳名称，为一款{product_type}写一段吸引人的营销文案：\n{best_name}"
        )
        | model
        | StrOutputParser()
    )

    async def run_pipeline(inputs: Dict[str, str]) -> Dict[str, str]:
        product_type = inputs["product_type"]
        product_names = await name_chain.ainvoke({"product_type": product_type})
        best_name = await select_chain.ainvoke({"product_names": product_names})
        marketing_copy = await marketing_chain.ainvoke(
            {"product_type": product_type, "best_name": best_name}
        )
        return {
            "product_names": product_names,
            "best_name": best_name,
            "marketing_copy": marketing_copy,
        }

    overall_chain = RunnableLambda(run_pipeline)

    print("\n顺序链执行流程:")
    print("  1. 生成产品名称")
    print("  2. 选择最佳名称")
    print("  3. 生成营销文案")

    result = await overall_chain.ainvoke({"product_type": "智能手表"})

    print("\n最终结果:")
    print(f"  产品名称列表: {result['product_names']}")
    print(f"  最佳名称: {result['best_name']}")
    print(f"  营销文案: {result['marketing_copy']}")


# ============================================================================
# 第三部分：复杂的数据转换链
# ============================================================================


def demo_transformation_chain():
    """
    使用 RunnableLambda 实现数据转换。

    旧版 TransformationChain 在部分新环境中不可用，
    因此这里改为直接使用 Runnable 进行等价演示。
    """
    print("\n" + "=" * 60)
    print("【3. Transformation 风格转换链】")
    print("=" * 60)

    def extract_keywords(inputs: Dict[str, str]) -> Dict[str, List[str]]:
        """从输入字典中的文本提取关键词。"""
        text = inputs["text"]
        stop_words = {"人工智能"}
        normalized = (
            text.replace("，", " ")
            .replace("。", " ")
            .replace("、", " ")
            .replace(",", " ")
            .replace("：", " ")
        )
        words = normalized.split()
        print(words)
        keywords = [
            w
            for w in words
            if any(stop_words == w for stop_words in stop_words) and len(w) > 1
        ]
        print(keywords)
        return {"keywords": keywords}

    transform_chain = RunnableLambda[Dict[str, str], Dict[str, List[str]]](
        extract_keywords
    )

    test_text = (
        "人工智能是当前最热门的技术领域之一，它包括机器学习和深度学习等方向，人工智能。"
    )
    result = transform_chain.invoke({"text": test_text})

    print(f"输入文本: {test_text}")
    print(f"提取的关键词: {result['keywords']}")


# ============================================================================
# 第四部分：Router 风格分发
# ============================================================================


async def demo_router_chain():
    """
    使用 RunnableLambda 模拟按问题类型路由。

    这样可以避免旧版 RouterChain API 在新版 LangChain 中不稳定或不可用的问题。
    """
    print("\n" + "=" * 60)
    print("【4. Router 风格分发】")
    print("=" * 60)

    physics_chain = (
        PromptTemplate.from_template(
            "你是一位物理学专家。请用通俗易懂的语言解释以下物理学问题：\n\n问题：{input}"
        )
        | model
        | StrOutputParser()
    )

    math_chain = (
        PromptTemplate.from_template(
            "你是一位数学家。请严谨地解答以下数学问题：\n\n问题：{input}"
        )
        | model
        | StrOutputParser()
    )

    history_chain = (
        PromptTemplate.from_template(
            "你是一位历史学家。请从历史角度分析以下问题：\n\n问题：{input}"
        )
        | model
        | StrOutputParser()
    )

    default_chain = (
        PromptTemplate.from_template("你是一个通用助手。请回答：\n{input}")
        | model
        | StrOutputParser()
    )

    def route_question(inputs: Dict[str, str]):
        text = inputs["input"]
        if any(keyword in text for keyword in ["量子", "物理", "引力", "光速"]):
            return physics_chain
        if any(keyword in text for keyword in ["为什么", "等于", "方程", "数学"]):
            return math_chain
        if any(keyword in text for keyword in ["秦始皇", "朝代", "历史", "战争"]):
            return history_chain
        return default_chain

    router_chain = RunnableLambda(route_question)

    questions = [
        "什么是量子纠缠？",
        "1+1为什么等于2？",
        "秦始皇统一六国的意义是什么？",
        "怎样提高学习效率？",
    ]

    print("路由链测试:")
    for q in questions:
        print(f"\n问题: {q}")
        selected_chain = route_question({"input": q})
        result = await selected_chain.ainvoke({"input": q})
        print(f"回答: {result[:100]}...")


# ============================================================================
# 第五部分：LCEL - LangChain Expression Language
# ============================================================================


async def demo_lcel():
    """
    LCEL (LangChain Expression Language)

    使用 | 运算符组合 Runnable 对象
    支持链式调用、并行执行、条件分支等
    """
    print("\n" + "=" * 60)
    print("【5. LCEL - LangChain 表达式语言】")
    print("=" * 60)

    basic_chain = (
        PromptTemplate.from_template("把{topic}描述成一句话")
        | model
        | StrOutputParser()
    )
    result = await basic_chain.ainvoke({"topic": "人工智能"})
    print(f"1. 基础管道: {result}")

    print("\n2. 并行处理:")
    text = "人工智能正在改变我们的生活方式"

    parallel_chain = RunnableParallel(
        summary=PromptTemplate.from_template("总结{text}") | model | StrOutputParser(),
        keywords=PromptTemplate.from_template("提取3个关键词：{text}")
        | model
        | StrOutputParser(),
        translation=PromptTemplate.from_template("翻译成英文：{text}")
        | model
        | StrOutputParser(),
    )

    parallel_result = await parallel_chain.ainvoke({"text": text})
    print(f"  总结: {parallel_result['summary']}")
    print(f"  关键词: {parallel_result['keywords']}")
    print(f"  英文: {parallel_result['translation']}")

    def create_summary_chain(format_type: str):
        """根据格式类型创建不同的摘要链"""
        if format_type == "short":
            template = "用一句话总结：{text}"
        elif format_type == "detailed":
            template = "详细总结，包括主要观点和细节：\n{text}"
        else:
            template = "总结：{text}"

        return PromptTemplate.from_template(template) | model | StrOutputParser()

    short_chain = create_summary_chain("short")
    detailed_chain = create_summary_chain("detailed")

    result_short = await short_chain.ainvoke({"text": text})
    result_detailed = await detailed_chain.ainvoke({"text": text})

    print(f"\n简短总结: {result_short}")
    print(f"详细总结: {result_detailed}")


# ============================================================================
# 第六部分：自定义 Chain
# ============================================================================


class CustomAnalysisChain:
    """
    自定义分析链示例。

    使用普通类封装 Runnable 创建逻辑，避免依赖旧版 LLMChain 继承体系。
    """

    def __init__(self, runnable, input_keys: List[str], output_keys: List[str]):
        self.runnable = runnable
        self.input_keys = input_keys
        self.output_keys = output_keys

    @classmethod
    def from_text_analysis(cls, llm):
        """从文本创建分析链。"""
        prompt = PromptTemplate.from_template(
            """分析以下文本，并提取信息：

文本：{text}

请提供：
1. 主题
2. 情感倾向（积极/消极/中性）
3. 关键信息
"""
        )
        runnable = prompt | llm | StrOutputParser()
        return cls(runnable=runnable, input_keys=["text"], output_keys=["analysis"])

    async def ainvoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        result = await self.runnable.ainvoke(inputs)
        return {"analysis": result}


async def demo_custom_chain():
    """
    使用自定义 Chain
    """
    print("\n" + "=" * 60)
    print("【6. 自定义 Chain】")
    print("=" * 60)

    chain = CustomAnalysisChain.from_text_analysis(llm=model)
    result = await chain.ainvoke({"text": "今天股市上涨了哦！"})

    print(result)

    print(f"自定义 Chain 输入变量: {chain.input_keys}")
    print(f"自定义 Chain 输出变量: {chain.output_keys}")


# ============================================================================
# 第七部分：实际应用 - 新闻处理流水线
# ============================================================================


async def demo_news_pipeline():
    """
    实际应用：新闻处理流水线

    流程：
    1. 生成新闻标题
    2. 生成新闻内容
    3. 提取关键信息
    4. 翻译成英文
    5. 生成摘要
    """
    print("\n" + "=" * 60)
    print("【7. 实际应用 - 新闻处理流水线】")
    print("=" * 60)

    title_chain = (
        PromptTemplate.from_template("为一个{event}事件生成一个吸引眼球的新闻标题")
        | model
        | StrOutputParser()
    )

    content_chain = (
        PromptTemplate.from_template("根据标题'{title}'写一篇200字左右的新闻报道")
        | model
        | StrOutputParser()
    )

    extract_chain = (
        PromptTemplate.from_template(
            """从以下新闻内容中提取关键信息，格式如下：
时间：<时间>
地点：<地点>
人物：<涉及的人物>
事件：<主要事件>

新闻内容：
{content}
"""
        )
        | model
        | StrOutputParser()
    )

    translate_chain = (
        PromptTemplate.from_template("将以下中文翻译成英文：\n{content}")
        | model
        | StrOutputParser()
    )

    summary_chain = (
        PromptTemplate.from_template("用50字以内总结以下新闻：\n{content}")
        | model
        | StrOutputParser()
    )

    async def run_news_pipeline(inputs: Dict[str, str]) -> Dict[str, str]:
        event = inputs["event"]
        title = await title_chain.ainvoke({"event": event})
        content = await content_chain.ainvoke({"title": title})
        key_info = await extract_chain.ainvoke({"content": content})
        english_content = await translate_chain.ainvoke({"content": content})
        summary = await summary_chain.ainvoke({"content": content})
        return {
            "title": title,
            "content": content,
            "key_info": key_info,
            "english_content": english_content,
            "summary": summary,
        }

    news_pipeline = RunnableLambda(run_news_pipeline)
    result = await news_pipeline.ainvoke({"event": "科技公司发布新产品"})

    print("\n最终结果:")
    print(f"标题: {result['title']}")
    print(f"摘要: {result['summary']}")
    print(f"关键信息: {result['key_info']}")
    print(f"英文内容: {result['english_content'][:100]}...")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """运行所有演示。"""
    print("=" * 60)
    print("LangChain Chain 教程")
    print("=" * 60)

    demo_llm_chain()
    await demo_llm_chain_invoke()
    # await demo_sequential_chain()
    # demo_transformation_chain()
    # await demo_router_chain()
    await demo_lcel()
    await demo_custom_chain()
    await demo_news_pipeline()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
