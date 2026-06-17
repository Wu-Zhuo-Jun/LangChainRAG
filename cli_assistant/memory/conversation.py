# -*- coding: utf-8 -*-
"""
=============================================================================
对话记忆模块
=============================================================================

管理对话历史的存储和加载。
"""

import json
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.memory import ConversationBufferMemory

import config


# ============================================================================
# 内存存储
# ============================================================================

# 内存中的对话历史
_memory_store: List[BaseMessage] = []


# ============================================================================
# 基础操作
# ============================================================================

def add_message(role: str, content: str):
    """
    添加消息到历史记录
    
    Args:
        role: 角色，"human" 或 "ai"
        content: 消息内容
    """
    if role == "human":
        _memory_store.append(HumanMessage(content=content))
    else:
        _memory_store.append(AIMessage(content=content))
    
    # 限制历史长度
    _trim_history()


def get_history() -> List[BaseMessage]:
    """
    获取对话历史
    
    Returns:
        消息列表
    """
    return _memory_store.copy()


def clear_history():
    """清空对话历史"""
    _memory_store.clear()
    print("对话历史已清空")


def _trim_history():
    """
    修剪历史记录，保持在限制长度内
    """
    global _memory_store
    if len(_memory_store) > config.MAX_HISTORY_LENGTH:
        _memory_store = _memory_store[-config.MAX_HISTORY_LENGTH:]


# ============================================================================
# 持久化存储
# ============================================================================

def save_to_file():
    """
    将对话历史保存到文件
    """
    try:
        # 转换为可序列化的格式
        data = []
        for msg in _memory_store:
            data.append({
                "type": msg.type,
                "content": msg.content
            })
        
        with open(config.CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"对话历史已保存到 {config.CONVERSATION_HISTORY_FILE}")
    
    except Exception as e:
        print(f"保存对话历史失败: {e}")


def load_from_file() -> bool:
    """
    从文件加载对话历史
    
    Returns:
        是否加载成功
    """
    try:
        if not config.CONVERSATION_HISTORY_FILE.exists():
            print("没有找到历史对话文件")
            return False
        
        with open(config.CONVERSATION_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 转换为消息对象
        _memory_store.clear()
        for item in data:
            if item["type"] == "human":
                _memory_store.append(HumanMessage(content=item["content"]))
            else:
                _memory_store.append(AIMessage(content=item["content"]))
        
        print(f"已加载 {len(_memory_store)} 条历史消息")
        return True
    
    except Exception as e:
        print(f"加载对话历史失败: {e}")
        return False


# ============================================================================
# Memory 对象
# ============================================================================

def get_memory() -> ConversationBufferMemory:
    """
    获取 ConversationBufferMemory 对象
    
    用于与 LangChain Agent 配合使用
    
    Returns:
        ConversationBufferMemory 实例
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        input_key="input",
    )
    
    # 加载现有历史
    for msg in _memory_store:
        if isinstance(msg, HumanMessage):
            memory.chat_memory.add_user_message(msg.content)
        else:
            memory.chat_memory.add_ai_message(msg.content)
    
    return memory


def update_memory(memory: ConversationBufferMemory):
    """
    更新内存中的历史记录
    
    用于同步 Agent 的 Memory 和本地存储
    
    Args:
        memory: Agent 的 ConversationBufferMemory 对象
    """
    global _memory_store
    _memory_store = list(memory.chat_memory.messages)
    _trim_history()


# ============================================================================
# 工具函数
# ============================================================================

def get_history_summary() -> str:
    """
    获取历史摘要
    
    Returns:
        摘要字符串
    """
    if not _memory_store:
        return "暂无对话历史"
    
    count = len(_memory_store)
    if count <= 2:
        return f"最近对话：{_memory_store[0].content[:30]}..."
    
    return f"共 {count} 条对话记录"


def print_history():
    """打印对话历史"""
    if not _memory_store:
        print("暂无对话历史")
        return
    
    print("\n" + "="*50)
    print("对话历史：")
    print("="*50)
    
    for i, msg in enumerate(_memory_store):
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"[{role}] {content}")
    
    print("="*50)
