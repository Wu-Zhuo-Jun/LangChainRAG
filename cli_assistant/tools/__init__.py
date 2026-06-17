# -*- coding: utf-8 -*-
"""
=============================================================================
工具模块
=============================================================================

定义智能助手可以使用的各种工具。
"""

from typing import List
from langchain_core.tools import tool, BaseTool
from datetime import datetime


# ============================================================================
# 天气工具
# ============================================================================

@tool
def get_weather(location: str) -> str:
    """
    查询指定城市的天气情况
    
    Args:
        location: 城市名称，如"北京"、"上海"、"广州"
    
    Returns:
        天气描述，包含温度、天气状况、空气质量等
    
    Example:
        >>> get_weather("北京")
        '北京天气：晴，温度25°C，湿度45%，风力2-3级，空气质量良好，适合户外活动'
    """
    # 模拟天气数据（实际项目中应调用真实天气 API）
    weather_db = {
        "北京": {
            "weather": "晴",
            "temp": "25°C",
            "humidity": "45%",
            "wind": "2-3级",
            "air_quality": "良好",
            "suggestion": "适合户外活动"
        },
        "上海": {
            "weather": "多云",
            "temp": "28°C",
            "humidity": "65%",
            "wind": "3-4级",
            "air_quality": "轻度污染",
            "suggestion": "建议戴口罩"
        },
        "广州": {
            "weather": "雷阵雨",
            "temp": "32°C",
            "humidity": "80%",
            "wind": "4-5级",
            "air_quality": "优",
            "suggestion": "建议带伞，谨慎出行"
        },
        "深圳": {
            "weather": "晴",
            "temp": "30°C",
            "humidity": "55%",
            "wind": "2级",
            "air_quality": "良",
            "suggestion": "紫外线较强，注意防晒"
        },
        "杭州": {
            "weather": "小雨",
            "temp": "24°C",
            "humidity": "70%",
            "wind": "2-3级",
            "air_quality": "良",
            "suggestion": "适合室内活动"
        },
        "成都": {
            "weather": "阴",
            "temp": "22°C",
            "humidity": "75%",
            "wind": "1-2级",
            "air_quality": "良",
            "suggestion": "天气凉爽舒适"
        },
        "武汉": {
            "weather": "晴",
            "temp": "29°C",
            "humidity": "50%",
            "wind": "2级",
            "air_quality": "良好",
            "suggestion": "适合外出"
        },
        "西安": {
            "weather": "多云转晴",
            "temp": "26°C",
            "humidity": "40%",
            "wind": "3级",
            "air_quality": "良",
            "suggestion": "天气宜人"
        },
    }
    
    # 尝试精确匹配
    if location in weather_db:
        info = weather_db[location]
        return (f"{location}天气：{info['weather']}，"
                f"温度{info['temp']}，"
                f"湿度{info['humidity']}，"
                f"风力{info['wind']}，"
                f"空气质量{info['air_quality']}。"
                f"建议：{info['suggestion']}")
    
    # 尝试模糊匹配
    for city, info in weather_db.items():
        if location in city or city in location:
            return (f"{city}天气：{info['weather']}，"
                    f"温度{info['temp']}，"
                    f"湿度{info['humidity']}，"
                    f"风力{info['wind']}。"
                    f"建议：{info['suggestion']}")
    
    return f"抱歉，暂不支持查询「{location}」的天气信息。当前支持的城市有：北京、上海、广州、深圳、杭州、成都、武汉、西安等。"


# ============================================================================
# 计算器工具
# ============================================================================

@tool
def calculate(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式，如 "2+3*4"、"100/5"、"(10+20)*3"
    
    Returns:
        计算结果字符串
    
    Example:
        >>> calculate("2+3*4")
        '2+3*4 = 14'
    """
    try:
        # 安全检查：只允许基本数学运算
        allowed_chars = set("0123456789+-*/().e ")
        if not all(c in allowed_chars or c.isalnum() for c in expression):
            return f"计算错误：表达式包含非法字符"
        
        # 执行计算
        result = eval(expression)
        
        # 格式化结果
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 10)
        
        return f"{expression} = {result}"
    
    except ZeroDivisionError:
        return "计算错误：除数不能为零"
    except SyntaxError:
        return "计算错误：表达式语法错误"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ============================================================================
# 搜索工具
# ============================================================================

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    搜索互联网获取信息
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数，默认5条
    
    Returns:
        搜索结果摘要
    
    Example:
        >>> search_web("Python教程", max_results=3)
        "关于'Python教程'的搜索结果：\\n1. Python官方文档...\\n2. 菜鸟教程Python...\\n3. 《Python编程》
    """
    # 模拟搜索结果（实际项目中应调用真实搜索 API）
    # 这里使用一个简单的模拟数据库
    search_db = {
        "python": [
            "Python 官方文档：https://docs.python.org",
            "Python 菜鸟教程：https://www.runoob.com/python3/",
            "Python 基础教程 - 廖雪峰的官方网站",
            "Python Cookbook 中文版",
        ],
        "langchain": [
            "LangChain 官方文档：https://docs.langchain.com",
            "LangChain 中文网：https://www.langchain.com.cn",
            "LangChain GitHub：https://github.com/langchain-ai/langchain",
        ],
        "ai": [
            "人工智能 (AI) 是计算机科学的一个分支",
            "AI 的主要应用领域：机器学习、自然语言处理、计算机视觉",
            "OpenAI: https://openai.com - AI 研究公司",
        ],
    }
    
    # 查找相关内容
    query_lower = query.lower()
    results = []
    
    # 精确匹配
    if query_lower in search_db:
        results = search_db[query_lower][:max_results]
    
    # 模糊匹配
    if not results:
        for key, values in search_db.items():
            if key in query_lower or query_lower in key:
                results.extend(values[:2])  # 最多添加2条
        results = results[:max_results]
    
    # 生成结果
    if results:
        result_text = f"关于「{query}」的搜索结果：\n"
        for i, r in enumerate(results, 1):
            result_text += f"{i}. {r}\n"
        return result_text.strip()
    else:
        return f"暂无「{query}」的相关搜索结果。建议换个关键词试试。"


# ============================================================================
# 时间和日期工具
# ============================================================================

@tool
def get_current_time() -> str:
    """
    获取当前日期和时间信息
    
    Returns:
        当前日期和时间的完整描述
    
    Example:
        >>> get_current_time()
        '今天是2024年1月15日，星期一，时间14:30:45'
    """
    now = datetime.now()
    
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    weekday = weekdays[now.weekday()]
    
    lunar_info = "农历日期需查表获取"  # 简化处理
    
    return (f"今天是{now.year}年{now.month}月{now.day}日，"
            f"星期{weekday}，"
            f"时间 {now.strftime('%H:%M:%S')}。"
            f"{lunar_info}")


# ============================================================================
# 百科问答工具
# ============================================================================

@tool
def query_knowledge(question: str) -> str:
    """
    查询百科知识
    
    Args:
        question: 知识问题，如"什么是人工智能"、"Python的优势"
    
    Returns:
        知识回答
    
    Example:
        >>> query_knowledge("什么是人工智能")
        "人工智能（AI）是..."
    """
    # 简单的知识库（实际项目中应使用真实百科 API）
    knowledge_db = {
        "人工智能": "人工智能（Artificial Intelligence，简称AI）是一门研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的新技术科学。它是计算机科学的一个分支，旨在生产出一种能以人类智能相似的方式做出反应的智能机器。",
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于1991年创建。特点是语法简洁、易于学习，拥有丰富的第三方库支持，广泛应用于 Web 开发、数据分析、人工智能、科学计算等领域。",
        "机器学习": "机器学习是人工智能的一个分支，专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。",
        "深度学习": "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。深度学习在计算机视觉、语音识别、自然语言处理等领域取得了突破性进展。",
    }
    
    question_lower = question.lower()
    
    # 精确匹配
    for key, answer in knowledge_db.items():
        if key in question_lower:
            return f"【{key}】\n{answer}"
    
    # 无法回答
    return f"抱歉，我目前没有「{question}」相关的知识储备。建议您换个问题，或者我可以帮您搜索相关信息。"


# ============================================================================
# 工具集合
# ============================================================================

def get_all_tools() -> List[BaseTool]:
    """
    获取所有可用的工具
    
    Returns:
        工具列表
    """
    return [
        get_weather,
        calculate,
        search_web,
        get_current_time,
        query_knowledge,
    ]


def get_tools_description() -> str:
    """
    获取所有工具的描述，用于 Prompt
    
    Returns:
        工具描述字符串
    """
    tools = get_all_tools()
    descriptions = []
    
    for t in tools:
        descriptions.append(f"- {t.name}: {t.description}")
    
    return "\n".join(descriptions)
