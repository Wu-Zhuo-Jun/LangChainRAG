# -*- coding: utf-8 -*-
"""
=============================================================================
Python 异步编程基础教程
=============================================================================

本文件讲解 Python 异步编程的核心概念，帮助你理解 async/await 语法，
为后续 FastAPI 和 LangChain 异步操作打下基础。

核心概念:
1. 同步 vs 异步：理解协程、事件循环、任务调度
2. async/await 语法：定义和调用异步函数
3. 并发执行：asyncio.gather 并行运行多个任务
4. 异步上下文管理：异步文件操作、数据库操作

=============================================================================
"""

import asyncio
import time
from typing import List, Coroutine, Any
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# 第一部分：同步 vs 异步基础对比
# ============================================================================


def sync_task(name: str, duration: float) -> str:
    """
    同步任务函数

    特点：
    - 使用 time.sleep() 会阻塞整个线程
    - 所有任务必须按顺序执行
    - 总耗时 = sum(所有任务耗时)

    Args:
        name: 任务名称
        duration: 任务执行时间（秒）

    Returns:
        任务完成消息
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行同步任务: {name}")
    time.sleep(duration)  # 同步阻塞，会卡住整个程序
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成同步任务: {name}")
    return f"同步任务 {name} 完成"


async def async_task(name: str, duration: float) -> str:
    """
    异步任务函数

    特点：
    - 使用 await asyncio.sleep() 只会暂停当前协程，不会阻塞其他协程
    - 多个任务可以并发执行
    - 总耗时 = max(最长任务耗时)

    Args:
        name: 任务名称
        duration: 任务执行时间（秒）

    Returns:
        任务完成消息
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行异步任务: {name}")
    await asyncio.sleep(duration)  # 异步暂停，释放控制权给事件循环
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成异步任务: {name}")
    return f"异步任务 {name} 完成"


def demo_sync_sequential():
    """
    演示：同步顺序执行多个任务

    假设有 3 个任务，每个耗时 1 秒
    顺序执行总耗时 = 1 + 1 + 1 = 3 秒
    """
    print("\n" + "=" * 60)
    print("【演示1】同步顺序执行")
    print("=" * 60)

    start = time.time()

    # 按顺序执行 3 个任务
    result1 = sync_task("任务A", 1.0)
    result2 = sync_task("任务B", 1.0)
    result3 = sync_task("任务C", 1.0)

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"结果: {result1}, {result2}, {result3}")


async def demo_async_concurrent():
    """
    演示：异步并发执行多个任务

    假设有 3 个任务，每个耗时 1 秒
    并发执行总耗时 = max(1, 1, 1) = 1 秒
    """
    print("\n" + "=" * 60)
    print("【演示2】异步并发执行")
    print("=" * 60)

    start = time.time()

    # 创建 3 个协程对象（注意：此时还未执行）
    tasks = [
        async_task("任务A", 1.0),
        async_task("任务B", 1.0),
        async_task("任务C", 1.0),
    ]

    # asyncio.gather 并发执行所有协程
    # 等待所有协程完成
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"结果: {results}")


# ============================================================================
# 第二部分：深入理解 async/await
# ============================================================================


async def await_syntax_explained():
    """
    async/await 语法详解
    """
    print("\n" + "=" * 60)
    print("【async/await 语法说明】")
    print("=" * 60)

    # 1. async def 定义异步函数
    #    - 调用时会返回一个协程对象（Coroutine），而不是直接执行
    #    - 必须通过 await 或 asyncio.run() 来执行

    coro = async_task("解释任务", 0.5)  # 返回协程对象，不执行函数体
    print(f"协程对象类型: {type[Any, Any, str](coro)}")

    # 2. await 暂停当前协程，等待另一个协程完成
    #    - await 可以获取协程的返回值
    #    - 等待期间，事件循环可以执行其他协程

    result = await coro  # 真正开始执行，并等待完成
    print(f"获取结果: {result}")

    # 3. 协程不能直接调用，必须通过事件循环执行
    #    asyncio.run() 创建事件循环并执行协程


# ============================================================================
# 第三部分：asyncio 常用 API
# ============================================================================


async def asyncio_api_examples():
    """
    asyncio 常用 API 示例
    """
    print("\n" + "=" * 60)
    print("【asyncio 常用 API】")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. asyncio.create_task() - 创建后台任务
    #    与直接 await 不同，create_task 会立即调度任务并在后台运行
    # -------------------------------------------------------------------------
    print("\n--- 1. asyncio.create_task() ---")

    async def background_job():
        await asyncio.sleep(2)
        return "后台任务完成"

    # 创建任务（立即调度）
    task = asyncio.create_task(background_job())

    # 可以同时做其他事情
    print("主线程: 正在执行其他工作...")
    await asyncio.sleep(0.5)
    print("主线程: 其他工作完成")

    # 等待后台任务完成
    result = await task
    print(f"后台任务结果: {result}")

    # -------------------------------------------------------------------------
    # 2. asyncio.wait_for() - 带超时等待
    #    如果超时，会抛出 asyncio.TimeoutError
    # -------------------------------------------------------------------------
    print("\n--- 2. asyncio.wait_for() 超时控制 ---")

    async def slow_task():
        await asyncio.sleep(3)
        return "慢任务完成"

    try:
        # 设置 1 秒超时
        result = await asyncio.wait_for(slow_task(), timeout=1.0)
        print(f"结果: {result}")
    except asyncio.TimeoutError:
        print("任务超时了！")

    # -------------------------------------------------------------------------
    # 3. asyncio.as_completed() - 按完成顺序处理结果
    #    当有多个任务时，可以立即处理已完成的而不等待全部完成
    # -------------------------------------------------------------------------
    print("\n--- 3. asyncio.as_completed() 按完成顺序 ---")

    async def varying_task(name: str, delay: float):
        await asyncio.sleep(delay)
        return f"{name} 完成"

    tasks = [
        asyncio.create_task(varying_task("快速", 0.5)),
        asyncio.create_task(varying_task("中速", 1.5)),
        asyncio.create_task(varying_task("慢速", 2.5)),
        asyncio.create_task(varying_task("快速", 0.5)),
    ]

    # 按完成顺序处理
    for completed_coro in asyncio.as_completed(tasks):
        result = await completed_coro
        print(f"-> {result}")

    # -------------------------------------------------------------------------
    # 4. asyncio.sleep() - 异步睡眠
    #    与 time.sleep() 不同，asyncio.sleep 不会阻塞线程
    # -------------------------------------------------------------------------
    print("\n--- 4. asyncio.sleep() vs time.sleep() ---")
    print("asyncio.sleep(): 异步暂停，可并发其他协程")
    print("time.sleep():    同步阻塞，整个线程暂停")


# ============================================================================
# 第四部分：异步生成器和异步上下文管理器
# ============================================================================


async def async_generators():
    """
    异步生成器示例

    使用 async for 遍历异步生成器
    """
    print("\n" + "=" * 60)
    print("【异步生成器】")
    print("=" * 60)

    async def async_range(start: int, end: int, delay: float = 0.1):
        """异步生成器：模拟逐个产生数据"""
        for i in range(start, end):
            await asyncio.sleep(delay)  # 模拟 IO 操作
            yield i

    # 使用 async for 遍历
    async for value in async_range(1, 6):
        print(f"收到数据: {value}")


class AsyncDatabase:
    """
    异步上下文管理器示例

    用于异步资源管理，如数据库连接、文件句柄等
    """

    def __init__(self, name: str):
        self.name = name
        self.connected = False

    async def connect(self):
        """模拟连接数据库"""
        await asyncio.sleep(0.5)  # 模拟网络延迟
        self.connected = True
        print(f"[{self.name}] 连接成功")

    async def disconnect(self):
        """模拟断开连接"""
        await asyncio.sleep(0.3)
        self.connected = False
        print(f"[{self.name}] 断开连接")

    async def query(self, sql: str) -> List[str]:
        """模拟执行查询"""
        if not self.connected:
            raise RuntimeError("未连接数据库")
        await asyncio.sleep(0.2)  # 模拟查询延迟
        return [f"结果行{i}" for i in range(3)]

    async def __aenter__(self):
        """进入上下文时调用"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时调用"""
        await self.disconnect()


async def async_context_manager():
    """
    异步上下文管理器示例

    使用 async with 自动管理资源
    """
    print("\n" + "=" * 60)
    print("【异步上下文管理器】")
    print("=" * 60)

    # 使用 async with 自动管理数据库连接
    async with AsyncDatabase("主数据库") as db:
        print(f"数据库已连接: {db.connected}")
        results = await db.query("SELECT * FROM users")
        print(f"查询结果: {results}")
        # 离开 with 块时自动断开连接

    print(f"退出后连接状态: {db.connected}")


# ============================================================================
# 第五部分：实际应用场景
# ============================================================================


@dataclass
class WeatherData:
    """天气数据模型"""

    city: str
    temperature: float
    condition: str


async def fetch_weather(city: str) -> WeatherData:
    """
    模拟异步获取天气数据

    实际项目中，这里会调用真实的天气 API
    """
    await asyncio.sleep(0.5)  # 模拟 API 延迟

    weather_db = {
        "北京": WeatherData("北京", 25.0, "晴"),
        "上海": WeatherData("上海", 28.0, "多云"),
        "广州": WeatherData("广州", 32.0, "雷阵雨"),
    }

    return weather_db.get(city, WeatherData(city, 20.0, "未知"))


async def fetch_multiple_weather(cities: List[str]) -> List[WeatherData]:
    """
    批量获取多个城市的天气

    使用 asyncio.gather 实现并发请求
    """
    print("\n" + "=" * 60)
    print("【实际应用】批量获取天气数据")
    print("=" * 60)

    start = time.time()

    # 创建所有协程
    tasks = [fetch_weather(city) for city in cities]

    # 并发执行
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    print(f"\n查询了 {len(cities)} 个城市，耗时 {elapsed:.2f} 秒")
    for weather in results:
        print(f"  {weather.city}: {weather.temperature}°C, {weather.condition}")

    return results


# ============================================================================
# 主函数入口
# ============================================================================


async def main():
    """
    主函数：运行所有演示
    """
    print("=" * 60)
    print("Python 异步编程基础教程")
    print("=" * 60)

    # 1. 同步顺序执行
    demo_sync_sequential()

    # 2. 异步并发执行
    await demo_async_concurrent()

    # 3. async/await 语法详解
    await await_syntax_explained()

    # 4. asyncio 常用 API
    await asyncio_api_examples()

    # 5. 异步生成器
    await async_generators()

    # 6. 异步上下文管理器
    await async_context_manager()

    # 7. 实际应用：批量获取天气
    await fetch_multiple_weather(["北京", "上海", "广州"])

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    """
    运行方式：
    python async_basics.py
    
    asyncio.run() 会：
    1. 创建新的事件循环
    2. 运行主协程直到完成
    3. 关闭事件循环
    """
    asyncio.run(main())
