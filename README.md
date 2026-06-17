# langChainStudy

一个用于学习 LangChain、FastAPI、Pydantic 和命令行 Agent 的示例项目集合。

## 项目内容

### 1. LangChain 示例
- `main.py`：基础模型调用与对话记忆示例
- `langchain_core/01_prompt_templates.py`：PromptTemplate、ChatPromptTemplate、MessagesPlaceholder 等用法
- `langchain_core/02_chains.py`：链式调用示例
- `memory/06_memory.py`：记忆相关示例
- `agents/05_react_agent.py`：ReAct Agent 示例
- `openai_functions/07_openai_functions.py`：OpenAI Functions / Tool Calling 示例
- `tools/weather_tool.py`：工具调用示例

### 2. FastAPI 示例
- `fastapi_project/main.py`：基础 FastAPI CRUD 示例
- `fastapi_project/advanced_models.py`：更复杂的请求体、响应模型、文件上传等示例

### 3. 命令行智能助手
- `cli_assistant/cli.py`：命令行入口
- `cli_assistant/agent/react_agent.py`：Agent 逻辑
- `cli_assistant/memory/conversation.py`：对话记忆
- `cli_assistant/tools/`：工具模块

## 环境要求

- Python 3.10+
- 建议使用虚拟环境

## 安装依赖

```bash
pip install -r requirements.txt
```

如果你想先创建虚拟环境，可以先执行：

```bash
python -m venv .venv
```

激活虚拟环境后再安装依赖。

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.\.venv\Scripts\activate.bat
```

## 运行示例

### 运行 FastAPI 示例

在 `fastapi_project` 目录下执行：

```bash
uvicorn main:app --reload
```

如果你在项目根目录运行，也可以指定模块路径：

```bash
uvicorn fastapi_project.main:app --reload
```

然后访问：

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### 运行 LangChain 示例

直接执行对应脚本，例如：

```bash
python main.py
```

或者：

```bash
python langchain_core/01_prompt_templates.py
```

### 运行命令行助手

```bash
python cli_assistant/cli.py
```

## 配置说明

部分示例需要 API Key。推荐使用环境变量，不要把密钥硬编码到代码里。

例如：

```powershell
$env:OPENAI_API_KEY="你的API_KEY"
```

如果你使用的是 DeepSeek 或其他兼容 OpenAI 接口的服务，请根据对应代码中的 `base_url` 和 `model_provider` 调整配置。

## 注意事项

- 当前项目中有部分示例更适合学习和演示，不一定是生产级写法。
- 建议把硬编码的密钥迁移到环境变量或 `.env` 文件中。
- 如果你要提交到 Git 仓库，建议补充 `.gitignore`，避免提交敏感信息和缓存文件。

## 依赖列表

本项目当前使用的核心依赖见 `requirements.txt`。

## 常备命令
<!-- Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -->
<!-- & d:/Code/langChainStudy/.venv/Scripts/Activate.ps1        -->
