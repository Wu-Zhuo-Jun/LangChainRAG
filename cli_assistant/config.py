# -*- coding: utf-8 -*-
"""
=============================================================================
配置文件
=============================================================================

集中管理所有配置项，便于统一修改。
"""

import os
from pathlib import Path


# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据存储目录
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 对话历史存储文件
CONVERSATION_HISTORY_FILE = DATA_DIR / "conversation_history.json"


# ============================================================================
# LLM 配置
# ============================================================================

# API 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-e6d2f16fbdd5462ea26a0d8202e843fc")
BASE_URL = "https://api.deepseek.com"

# 模型配置
MODEL_NAME = "deepseek-chat"
MODEL_TEMPERATURE = 0.7
MODEL_MAX_TOKENS = 2000


# ============================================================================
# Agent 配置
# ============================================================================

# Agent 行为配置
MAX_ITERATIONS = 10  # Agent 最大迭代次数
MAX_EXECUTION_TIME = 60  # 最大执行时间（秒）
VERBOSE = True  # 是否打印详细日志

# 对话历史配置
MAX_HISTORY_LENGTH = 50  # 最大保存历史消息数


# ============================================================================
# Tool 配置
# ============================================================================

# 天气 API（可替换为真实 API）
WEATHER_API_ENABLED = True

# 搜索配置
SEARCH_ENABLED = True
SEARCH_MAX_RESULTS = 5

# 计算器配置
CALCULATOR_SAFE_MODE = True  # 是否启用安全模式


# ============================================================================
# UI 配置
# ============================================================================

# 控制台样式
STYLE_USER = "[bold yellow]"  # 用户输入样式
STYLE_ASSISTANT = "[bold cyan]"  # 助手输出样式
STYLE_ERROR = "[bold red]"  # 错误信息样式
STYLE_SUCCESS = "[bold green]"  # 成功信息样式
STYLE_INFO = "[bold blue]"  # 信息样式
STYLE_SYSTEM = "[bold magenta]"  # 系统消息样式


# ============================================================================
# 工具函数
# ============================================================================

def get_all_config():
    """获取所有配置"""
    return {
        "model": {
            "name": MODEL_NAME,
            "temperature": MODEL_TEMPERATURE,
            "max_tokens": MODEL_MAX_TOKENS,
        },
        "agent": {
            "max_iterations": MAX_ITERATIONS,
            "verbose": VERBOSE,
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
        }
    }
