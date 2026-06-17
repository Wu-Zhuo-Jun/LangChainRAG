# -*- coding: utf-8 -*-
"""
=============================================================================
LLM 模块
=============================================================================

负责 LLM 模型的初始化和配置。
"""

import os
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

import config


# ============================================================================
# 全局模型实例（单例模式）
# ============================================================================

_llm_instance: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """
    获取 LLM 实例（单例）
    
    Returns:
        初始化好的 LLM 模型实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = _create_llm()
    
    return _llm_instance


def _create_llm() -> BaseChatModel:
    """
    创建 LLM 模型实例
    
    Returns:
        配置好的 ChatModel 实例
    """
    print(f"正在初始化 LLM 模型: {config.MODEL_NAME}")
    
    # 设置 API Key
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    
    # 初始化模型
    llm = init_chat_model(
        model=config.MODEL_NAME,
        model_provider="deepseek",
        api_key=config.OPENAI_API_KEY,
        base_url=config.BASE_URL,
        temperature=config.MODEL_TEMPERATURE,
    )
    
    print("LLM 模型初始化完成！")
    
    return llm


def reset_llm():
    """
    重置 LLM 实例
    
    某些情况下可能需要重新初始化模型
    """
    global _llm_instance
    _llm_instance = None
    print("LLM 实例已重置")
