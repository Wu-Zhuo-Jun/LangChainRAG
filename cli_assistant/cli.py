# -*- coding: utf-8 -*-
"""
=============================================================================
命令行交互界面
=============================================================================

提供友好的命令行交互界面。
"""

import sys
import asyncio
from typing import Optional

# 尝试导入 rich 库美化输出
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("提示: 安装 rich 库可以获得更好的界面体验")
    print("  pip install rich")

import config
from agent import react_agent as agent_module
from memory import conversation as memory_module


# ============================================================================
# 控制台初始化
# ============================================================================

if RICH_AVAILABLE:
    console = Console()
else:
    # 简单的控制台输出
    class SimpleConsole:
        def print(self, text, style=None, **kwargs):
            print(text)
        
        def input(self, prompt=""):
            return input(prompt)
    
    console = SimpleConsole()


# ============================================================================
# 欢迎界面
# ============================================================================

def print_welcome():
    """打印欢迎信息"""
    welcome_text = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║           欢迎使用智能助手 (Smart Assistant)           ║
    ║                                                       ║
    ║   我可以帮你：                                        ║
    ║   - 回答各种问题                                       ║
    ║   - 查询天气信息                                       ║
    ║   - 执行数学计算                                       ║
    ║   - 搜索网络信息                                       ║
    ║   - 了解更多知识                                       ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            welcome_text,
            title="[bold cyan]智能助手[/bold cyan]",
            border_style="cyan"
        ))
    else:
        print(welcome_text)


def print_help():
    """打印帮助信息"""
    help_text = """
    ╔═══════════════════════════════════════════════════════╗
    ║                    命令帮助                           ║
    ╠═══════════════════════════════════════════════════════╣
    ║  /help 或 ?       - 显示帮助信息                      ║
    ║  /history         - 查看对话历史                       ║
    ║  /clear           - 清空对话历史                       ║
    ║  /save            - 保存对话历史                       ║
    ║  /load            - 加载历史对话                       ║
    ║  /info            - 显示系统信息                       ║
    ║  /reset           - 重置 Agent                        ║
    ║  /exit 或 quit    - 退出程序                          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    
    if RICH_AVAILABLE:
        console.print(Panel.fit(help_text, title="帮助", border_style="green"))
    else:
        print(help_text)


def print_info():
    """打印系统信息"""
    info = agent_module.get_agent_info()
    
    if RICH_AVAILABLE:
        table = Table(title="系统信息")
        table.add_column("配置项", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("模型", config.MODEL_NAME)
        table.add_row("温度", str(config.MODEL_TEMPERATURE))
        table.add_row("最大迭代", str(info["max_iterations"]))
        table.add_row("工具数量", str(info["tools_count"]))
        table.add_row("可用工具", ", ".join(info["tools"]))
        table.add_row("对话历史", memory_module.get_history_summary())
        
        console.print(table)
    else:
        print("\n系统信息:")
        print(f"  模型: {config.MODEL_NAME}")
        print(f"  温度: {config.MODEL_TEMPERATURE}")
        print(f"  最大迭代: {info['max_iterations']}")
        print(f"  工具数量: {info['tools_count']}")
        print(f"  可用工具: {', '.join(info['tools'])}")
        print(f"  对话历史: {memory_module.get_history_summary()}")


# ============================================================================
# 主循环
# ============================================================================

async def main_loop():
    """
    主对话循环
    """
    # 打印欢迎信息
    print_welcome()
    
    # 尝试加载历史
    memory_module.load_from_file()
    
    # 获取 Agent
    try:
        agent_module.get_agent()
    except Exception as e:
        console.print(f"[red]Agent 初始化失败: {e}[/red]" if RICH_AVAILABLE else f"Agent 初始化失败: {e}")
        return
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = console.input("\n[用户] " if RICH_AVAILABLE else "\n用户: ")
            
            # 去掉首尾空格
            user_input = user_input.strip()
            
            # 空输入
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ["/exit", "/quit", "exit", "quit", "退出"]:
                console.print("[green]感谢使用，再见！[/green]" if RICH_AVAILABLE else "感谢使用，再见！")
                break
            
            elif user_input.lower() in ["/help", "/?", "help", "?"]:
                print_help()
                continue
            
            elif user_input.lower() == "/history":
                memory_module.print_history()
                continue
            
            elif user_input.lower() == "/clear":
                memory_module.clear_history()
                console.print("[green]对话历史已清空[/green]" if RICH_AVAILABLE else "对话历史已清空")
                continue
            
            elif user_input.lower() == "/save":
                memory_module.save_to_file()
                continue
            
            elif user_input.lower() == "/load":
                memory_module.load_from_file()
                continue
            
            elif user_input.lower() == "/info":
                print_info()
                continue
            
            elif user_input.lower() == "/reset":
                agent_module.reset_agent()
                console.print("[green]Agent 已重置[/green]" if RICH_AVAILABLE else "Agent 已重置")
                continue
            
            # 正常对话
            console.print("[dim]思考中...[/dim]" if RICH_AVAILABLE else "思考中...", end="")
            
            # 调用 Agent
            response = await agent_module.run_agent_async(user_input)
            
            # 保存对话
            memory_module.add_message("human", user_input)
            memory_module.add_message("ai", response)
            
            # 显示响应
            print("\n")
            if RICH_AVAILABLE:
                console.print(Panel(
                    response,
                    title="[bold cyan]助手[/bold cyan]",
                    border_style="cyan"
                ))
            else:
                print("="*50)
                print(f"助手: {response}")
                print("="*50)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]检测到 Ctrl+C，是否退出？[/yellow]" if RICH_AVAILABLE else "\n检测到 Ctrl+C，是否退出？")
            confirm = console.input("输入 y 确认退出: " if RICH_AVAILABLE else "输入 y 确认退出: ")
            if confirm.lower() == "y":
                break
        
        except Exception as e:
            console.print(f"[red]发生错误: {e}[/red]" if RICH_AVAILABLE else f"发生错误: {e}")


# ============================================================================
# 同步包装
# ============================================================================

def main():
    """
    入口函数
    """
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()
