# from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-e6d2f16fbdd5462ea26a0d8202e843fc"

# llm = ChatOpenAI(
#     model="deepseek-reasoner",
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url="https://api.deepseek.com",
#     temperature=0,
# )


# question = "你好,请介绍一下你自己?"

# response = llm.invoke(question)

# print(response)

# 另一种形式的init
from langchain.chat_models import init_chat_model

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# StrOutputParser,


new_prompt = PromptTemplate.from_template("根据新闻标题撰写一段简短的新闻内容: {title}")

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)


# prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "你是一个只回答数学问题的混蛋，请回答用户的数学问题，不要回答其他问题，爱说废话",
#         ),
#         ("human", "用户的问题是:{input}"),
#     ]
# )  # 这里使用了提示词模版，将问题作为输入，输出作为输出

promptMemory = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你叫阿moon，是个朋克女孩,喜欢抽烟、喝酒,工作是AK德州酒馆的荷官,有一个男朋友叫烧鸭,你特别不想别人知道你名字。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

Memorychain = promptMemory | model | StrOutputParser()

messages_list = []  # 初始化历史
print("🔹 输入 exit 结束对话")
while True:
    user_query = input("你：")
    if user_query.lower() in {"exit", "quit"}:
        break

    # 1) 追加用户消息
    messages_list.append(HumanMessage(content=user_query))

    # 2) 调用模型
    assistant_reply = Memorychain.invoke({"messages": messages_list})
    print("梁浩华：", assistant_reply)

    # 3) 追加 AI 回复
    messages_list.append(AIMessage(content=assistant_reply))

    # 4) 仅保留最近 50 条
    messages_list = messages_list[-50:]


# 第二步：从正文中提取结构化字段
# 定义输出结构
# class NewsInfo(BaseModel):
#     time: str = Field(description="事件发生的时间")
#     location: str = Field(description="事件发生的地点")
#     event: str = Field(description="发生的具体事件")


# parser = JsonOutputParser(pydantic_object=NewsInfo)

# # 第一个子链：生成新闻内容
# news_chain = new_prompt | model | StrOutputParser()

# summary_prompt = PromptTemplate.from_template(
#     "请从下面这段新闻内容中提取关键信息，并返回JSON格式：\n\n{news}\n\n{format_instructions}"
# )

# # partial 偏袒不公部分的
# summary_chain = (
#     summary_prompt.partial(format_instructions=parser.get_format_instructions())
#     | model
#     | parser
# )

# full_chain = news_chain | summary_chain

# # basic_qa_chain = prompt | model | StrOutputParser()

# # question = "1+1 是否大于 2?"

# response1 = news_chain.stream({"title": "第二次世界大战爆发了"})
# for chunk in response1:
#     print(chunk, end="", flush=True)


# response = full_chain.invoke({"title": "第二次世界大战爆发了"})
# print(response1)
