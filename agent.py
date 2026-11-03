"""最小 Agent：模型自己决定调用哪个工具。

相比 v1 的改动（只有 3 处）：
  1. 新增一个工具 search_docs —— 把 RAG 封装成工具
  2. TOOLS 里加上它的说明书
  3. TOOL_MAP 里加上它的实现

运行：
    python agent.py
"""

import json
import os

# from llm import chat, client
from llm import chat, chat_raw
from rag import find_hybrid

MODEL = "deepseek-flash"
MAX_STEPS = 8


# ══════════════════════════════════════════
# 1. 工具的真实实现
# ══════════════════════════════════════════

def list_files(dir: str) -> str:
    """列出目录下的文件名。"""
    return "\n".join(sorted(os.listdir(dir)))


def read_file(path: str, lines: int = 20) -> str:
    """读取文本文件的前若干行。"""
    with open(path, encoding="utf-8") as f:
        return "".join(f.readlines()[:lines])


def search_docs(question: str) -> str:
    """在本地文档库里检索并回答（就是把昨天的 RAG 包一层）。"""
    found = find_hybrid(question)
    context = "\n\n---\n\n".join(found)
    return chat([
        {"role": "system", "content":
            "只根据资料回答。先摘录原文，再给结论。资料里没有就说「资料里没有」。"},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ])


TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "search_docs": search_docs,
}


# ══════════════════════════════════════════
# 2. 工具的说明书
# ══════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出某个目录下的文件名",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {"type": "string", "description": "目录路径，例如 data"}
                },
                "required": ["dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文本文件的前若干行",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径，例如 data/xxx.md"},
                    "lines": {"type": "integer", "description": "读多少行，默认 20"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "在本地 DeepSeek 官方文档库中检索并回答问题。"
                "涉及 DeepSeek 模型名称、价格、并发限制、错误码、token 计算、API 调用方式时用它。"
                "闲聊、自我介绍、通用知识不需要用它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要查询的问题"}
                },
                "required": ["question"],
            },
        },
    },
]


# ══════════════════════════════════════════
# 3. Agent 循环
# ══════════════════════════════════════════

def run_agent(question: str, max_steps: int = MAX_STEPS) -> str:
    messages = [
        {"role": "system", "content":
            "你能调用工具获取信息。需要事实、数据、文档内容时先调工具，不要凭记忆回答。"
            "闲聊和常识问题可以直接答。"},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        # resp = client.chat.completions.create(
        #     model=MODEL, messages=messages, tools=TOOLS
        # )
        # msg = resp.choices[0].message
        msg = chat_raw(messages, tools=TOOLS)

        if not msg.tool_calls:
            print(f"  [步骤{step}] 模型决定：直接回答（没调工具）")
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in msg.tool_calls
            ],
        })

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  [步骤{step}] 调用 {name}({args})")

            try:
                result = TOOL_MAP[name](**args)
            except Exception as e:
                result = f"工具执行失败：{type(e).__name__}: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "达到最大步数，强制结束"


if __name__ == "__main__":
    questions = [
        "你好，你能做什么？",                                     # 不该调工具
        "deepseek-v4-pro 百万 tokens 输出的价格是多少？",          # 应该调 search_docs
        "data 目录下有几个文件？",                                 # 应该调 list_files
    ]

    for q in questions:
        print(f"\n{'=' * 56}\n问：{q}\n{'=' * 56}")
        print("答：", run_agent(q))
