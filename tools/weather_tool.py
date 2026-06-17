# -*- coding: utf-8 -*-
"""
=============================================================================
LangChain Tool 教程
=============================================================================

Tool（工具）是 Agent 与外部世界交互的桥梁。

Tool 的核心概念：
1. @tool 装饰器 - 最简单的 Tool 定义方式
2. create Tool - 手动创建 Tool
3. StructuredTool - 支持复杂参数的 Tool
4. Tool 的属性（name, description, args_schema）
5. Tool 的绑定和调用

=============================================================================
"""

import os
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool, BaseTool

# from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.utils.function_calling import convert_to_openai_function


# ============================================================================
# 环境配置
os.environ["DEEPSEEK_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
# ============================================================================


# ============================================================================
# 第一部分：@tool 装饰器基础
# ============================================================================


def demo_basic_tool():
    """
    使用 @tool 装饰器创建简单的 Tool

    这是最简洁的 Tool 定义方式
    """
    print("\n" + "=" * 60)
    print("【1. @tool 装饰器基础】")
    print("=" * 60)

    # 使用 @tool 装饰器
    @tool
    def get_weather(location: str) -> str:
        """查询指定城市的天气情况

        Args:
            location: 城市名称，如"北京"、"Shanghai"

        Returns:
            天气描述字符串，包含温度和天气状况
        """
        weather_db = {
            "北京": "晴，25°C，空气质量良好",
            "上海": "多云，28°C，有轻度污染",
            "广州": "雷阵雨，32°C，请带伞出行",
            "深圳": "晴，30°C，紫外线较强",
            "杭州": "小雨，24°C，适合室内活动",
        }

        result = weather_db.get(location, f"抱歉，暂不支持查询 {location} 的天气")
        return result

    # 查看 Tool 的属性
    print(f"\nTool 名称: {get_weather.name}")
    print(f"Tool 描述: {get_weather.description}")
    print(f"Tool 参数: {get_weather.args}")
    print(f"Tool 返回值: {get_weather.args_schema}")

    # 直接调用 Tool
    result = get_weather.invoke({"location": "北京"})
    print(f"\n调用结果: {result}")

    # 使用字符串参数（简单参数）
    result2 = get_weather.invoke("上海")
    print(f"字符串参数调用: {result2}")


def demo_tool_with_multiple_args():
    """
    Tool 多个参数示例
    """
    print("\n" + "=" * 60)
    print("【1.2 多参数 Tool】")
    print("=" * 60)

    @tool
    def calculate_route(start: str, end: str, mode: str = "driving") -> str:
        """计算两点之间的路线

        Args:
            start: 起始地点
            end: 目标地点
            mode: 出行方式，可选 "driving", "walking", "cycling"

        Returns:
            路线描述，包含距离和预计时间
        """
        routes = {
            ("天安门", "故宫", "driving"): "1.2公里，约5分钟",
            ("天安门", "故宫", "walking"): "1.2公里，约15分钟",
            ("天安门", "故宫", "cycling"): "1.2公里，约8分钟",
        }

        key = (start, end, mode)
        result = routes.get(key, f"从{start}到{end}的{mode}路线正在规划中...")

        return f"🚗 出行方式: {mode}\n📍 路线: {result}"

    print(f"Tool 参数: {calculate_route.args}")

    # 调用
    result = calculate_route.invoke(
        {"start": "天安门", "end": "故宫", "mode": "walking"}
    )
    print(f"\n调用结果:\n{result}")


# ============================================================================
# 第二部分：StructuredTool - 复杂参数
# ============================================================================


def demo_structured_tool():
    """
    StructuredTool 用于定义有复杂参数结构的 Tool

    使用 Pydantic 模型定义参数
    """
    print("\n" + "=" * 60)
    print("【2. StructuredTool - 结构化工具】")
    print("=" * 60)

    # 定义参数模型
    class SearchInput(BaseModel):
        query: str = Field(description="搜索关键词")
        max_results: int = Field(default=5, description="最大返回结果数")
        source: Optional[str] = Field(
            default=None, description="信息来源，如 'news', 'blog', 'wiki'"
        )

    def search_web(
        query: str, max_results: int = 5, source: Optional[str] = None
    ) -> str:
        source_text = f"，来源: {source}" if source else ""
        return f"关于'{query}'的搜索结果，共 {max_results} 条{source_text}"

    web_search = StructuredTool.from_function(
        func=search_web,
        name="web_search",
        description="使用搜索引擎查询相关信息",
        args_schema=SearchInput,
    )

    print(f"\nTool 名称: {web_search.name}")
    print(f"Tool 描述: {web_search.description}")
    print(f"参数模型: {web_search.args_schema}")

    # 使用 Pydantic 模型构造输入，再转为字典调用
    search_input = SearchInput(query="Python教程", max_results=3, source="blog")
    result = web_search.invoke(search_input.model_dump())
    print(f"\n调用结果: {result}")

    # 使用字典调用
    result2 = web_search.invoke({"query": "LangChain 入门", "max_results": 10})
    print(f"字典调用: {result2}")


# ============================================================================
# 第三部分：BaseTool - 自定义 Tool 类
# ============================================================================


def demo_custom_base_tool():
    """
    继承 BaseTool 创建自定义 Tool

    适用于需要更复杂逻辑的工具
    """
    print("\n" + "=" * 60)
    print("【3. BaseTool 自定义类】")
    print("=" * 60)

    class DatabaseQueryTool(BaseTool):
        """
        数据库查询工具示例

        模拟一个数据库查询操作
        """

        name: str = "database_query"
        description: str = """执行数据库查询操作
        
        输入：
        - query: SQL 查询语句
        - limit: 返回结果数量限制
        
        返回：
        - 查询结果列表
        """

        # 定义参数模型
        args_schema: Type[BaseModel] = DatabaseQueryInput

        def _run(self, query: str, limit: int = 10, run_manager=None) -> str:
            """
            同步执行方法

            Args:
                query: SQL 查询语句
                limit: 返回结果数量限制
                run_manager: 回调管理器

            Returns:
                查询结果
            """
            # 模拟数据库查询
            mock_results = [
                {"id": 1, "name": "张三", "age": 25},
                {"id": 2, "name": "李四", "age": 30},
                {"id": 3, "name": "王五", "age": 28},
            ]

            # 模拟执行查询
            result = f"执行查询: {query}\n"
            result += f"返回 {min(limit, len(mock_results))} 条记录:\n"
            for i, row in enumerate(mock_results[:limit]):
                result += f"  {i + 1}. {row}\n"

            return result

        async def _arun(self, query: str, limit: int = 10, run_manager=None) -> str:
            """
            异步执行方法

            实际项目中数据库操作通常是异步的
            """
            return self._run(query, limit, run_manager)

    # 创建工具实例
    db_tool = DatabaseQueryTool()

    print(f"Tool 名称: {db_tool.name}")
    print(f"Tool 描述: {db_tool.description}")

    # 调用
    result = db_tool.invoke({"query": "SELECT * FROM users WHERE age > 25", "limit": 2})
    print(f"\n执行结果:\n{result}")


class DatabaseQueryInput(BaseModel):
    """数据库查询工具的输入模型"""

    query: str = Field(description="SQL 查询语句")
    limit: int = Field(default=10, description="返回结果数量限制")


# ============================================================================
# 第四部分：Tool 的属性详解
# ============================================================================


def demo_tool_attributes():
    """
    Tool 属性详解

    每个 Tool 都有以下关键属性：
    1. name: 工具名称（用于调用）
    2. description: 工具描述（帮助 LLM 理解何时使用）
    3. args/args_schema: 参数定义
    4. return_schema: 返回值定义
    """
    print("\n" + "=" * 60)
    print("【4. Tool 属性详解】")
    print("=" * 60)

    @tool
    def search_products(
        category: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        keyword: Optional[str] = None,
    ) -> List[dict]:
        """搜索商品列表

        Args:
            category: 商品分类，如 "电子产品"、"图书"、"服装"
            min_price: 最低价格（可选）
            max_price: 最高价格（可选）
            keyword: 搜索关键词（可选）

        Returns:
            商品列表，每项包含 id, name, price, description
        """
        # 模拟商品数据
        products = [
            {"id": 1, "name": "iPhone 15", "price": 999, "category": "电子产品"},
            {"id": 2, "name": "MacBook Pro", "price": 1999, "category": "电子产品"},
            {"id": 3, "name": "Python编程", "price": 79, "category": "图书"},
            {"id": 4, "name": "T恤", "price": 99, "category": "服装"},
        ]

        # 过滤
        filtered = products

        if category:
            filtered = [p for p in filtered if p["category"] == category]

        if min_price:
            filtered = [p for p in filtered if p["price"] >= min_price]

        if max_price:
            filtered = [p for p in filtered if p["price"] <= max_price]

        if keyword:
            filtered = [p for p in filtered if keyword.lower() in p["name"].lower()]

        return filtered

    print("\n--- Tool 属性 ---")
    print(f"名称 (name): {search_products.name}")
    print(f"描述 (description):\n  {search_products.description}")
    print(f"\n参数 (args): {search_products.args}")
    print(f"\n参数 Schema: {search_products.args_schema}")

    # 调用示例
    result = search_products.invoke({"category": "电子产品", "min_price": 500})
    print(f"\n调用结果: {result}")


# ============================================================================
# 第五部分：Tool 的 OpenAI 格式转换
# ============================================================================


def demo_tool_to_openai():
    """
    将 LangChain Tool 转换为 OpenAI 函数调用格式

    使用 convert_to_openai_function 转换
    """
    print("\n" + "=" * 60)
    print("【5. Tool 转 OpenAI 格式】")
    print("=" * 60)

    @tool
    def get_weather(location: str) -> str:
        """查询指定城市的天气"""
        return "晴，25°C"

    @tool
    def search_news(keyword: str, limit: int = 5) -> str:
        """搜索新闻

        Args:
            keyword: 搜索关键词
            limit: 返回数量限制
        """
        return f"关于'{keyword}'的新闻..."

    tools = [get_weather, search_news]

    print("\n原始 Tool 信息:")
    for t in tools:
        print(f"\n  名称: {t.name}")
        print(f"  描述: {t.description}")
        print(f"  参数: {t.args}")

    # 转换为 OpenAI 格式
    print("\n--- 转换为 OpenAI 函数格式 ---")
    for t in tools:
        openai_func = convert_to_openai_function(t)
        print(f"\n{openai_func['name']}:")
        print(f"  {openai_func}")


# ============================================================================
# 第六部分：Tool 的实际应用
# ============================================================================


def demo_tools_in_action():
    """
    Tool 的实际应用场景
    """
    print("\n" + "=" * 60)
    print("【6. Tool 实际应用场景】")
    print("=" * 60)

    # 1. 计算器工具
    @tool
    def calculator(expression: str) -> str:
        """执行数学计算

        Args:
            expression: 数学表达式，如 "2+3*4" 或 "10/2"

        Returns:
            计算结果
        """
        try:
            # 注意：实际应用中应该使用安全的计算方法
            # 这里仅用于演示
            result = eval(expression)  # 简化演示
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    # 2. 翻译工具
    @tool
    def translate(
        text: str, source_lang: str = "中文", target_lang: str = "英文"
    ) -> str:
        """翻译文本

        Args:
            text: 要翻译的文本
            source_lang: 源语言
            target_lang: 目标语言
        """
        translations = {
            ("你好", "中文", "英文"): "Hello",
            ("谢谢", "中文", "英文"): "Thank you",
            ("再见", "中文", "英文"): "Goodbye",
        }

        key = (text, source_lang, target_lang)
        return translations.get(key, f"[翻译: {text} -> {target_lang}]")

    # 3. 日程管理工具
    @tool
    def schedule_event(
        title: str, date: str, time: str, participants: List[str] = None
    ) -> str:
        """创建日程事件

        Args:
            title: 事件标题
            date: 日期 (YYYY-MM-DD)
            time: 时间 (HH:MM)
            participants: 参与者列表
        """
        participants = participants or []
        return f"""📅 日程已创建:
  标题: {title}
  日期: {date}
  时间: {time}
  参与者: {", ".join(participants) if participants else "无"}
"""

    print("\n1. 计算器工具:")
    print(f"   2+3*4 = {calculator.invoke('2+3*4')}")

    print("\n2. 翻译工具:")
    print(f"   你好 -> {translate.invoke({'text': '你好', 'target_lang': '英文'})}")

    print("\n3. 日程管理工具:")
    print(
        schedule_event.invoke(
            {
                "title": "团队会议",
                "date": "2024-01-15",
                "time": "14:00",
                "participants": ["张三", "李四"],
            }
        )
    )


# ============================================================================
# 第七部分：Tool 绑定到模型
# ============================================================================


async def demo_bind_tools_to_model():
    """
    将 Tool 绑定到 LLM 模型

    模型将学会在适当时机调用工具
    """
    print("\n" + "=" * 60)
    print("【7. Tool 绑定到模型】")
    print("=" * 60)

    from langchain.chat_models import init_chat_model

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未检测到 API Key。请先设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    # 创建模型
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.7,
    )

    # 定义工具
    @tool
    def get_weather(location: str) -> str:
        """查询指定城市的天气"""
        return "晴，25°C"

    @tool
    def get_time(city: str) -> str:
        """查询指定城市当前时间"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools = [get_weather, get_time]

    # 绑定工具到模型
    model_with_tools = model.bind_tools(tools)

    print(f"已绑定 {len(tools)} 个工具到模型")
    print(f"工具列表: {[t.name for t in tools]}")

    # 调用（模型可能会决定调用工具）
    response = await model_with_tools.ainvoke("北京现在天气如何？")

    print(f"\n模型响应:")
    print(f"  类型: {type(response)}")
    print(f"  内容: {response}")

    # 检查是否有工具调用
    if hasattr(response, "tool_calls"):
        if len(response.tool_calls) > 0:
            print(f"\n模型决定调用工具:")
            for call in response.tool_calls:
                print(f"  函数: {call['name']}")
                print(f"  参数: {call['args']}")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """
    运行所有演示
    """
    print("=" * 60)
    print("LangChain Tool 教程")
    print("=" * 60)

    # 1. @tool 基础
    demo_basic_tool()
    demo_tool_with_multiple_args()

    # 2. StructuredTool
    demo_structured_tool()

    # 3. BaseTool 自定义
    demo_custom_base_tool()

    # 4. Tool 属性
    demo_tool_attributes()

    # 5. Tool 转 OpenAI 格式
    demo_tool_to_openai()

    # 6. 实际应用
    demo_tools_in_action()

    # 7. Tool 绑定模型
    await demo_bind_tools_to_model()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
