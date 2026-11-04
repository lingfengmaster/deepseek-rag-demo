"""最小 Agent：模型自己决定调用哪个工具。

v4：新增 search_code 工具 —— 代码库和文档库两套知识，由模型自己选。

运行：
    python agent.py
"""

import json
import os
from pathlib import Path

from code_rag import build_cards, build_index, load_cache, save_cache, search as search_cards
from llm import chat, chat_raw
from rag import find_hybrid

MODEL = "deepseek-flash"
MAX_STEPS = 8

ROOT = Path(__file__).resolve().parent

BLOCKED_NAMES = {".env", ".git", ".ssh", "id_rsa", "id_ed25519", ".credentials"}
BLOCKED_SUFFIXES = {".pem", ".key", ".p12"}


# ══════════════════════════════════════════
# 0. 安全层
# ══════════════════════════════════════════

def safe_path(raw: str) -> Path:
    if not raw or not isinstance(raw, str):
        raise PermissionError("路径不能为空")

    target = (ROOT / raw).resolve()

    if not target.is_relative_to(ROOT):
        raise PermissionError(f"拒绝访问项目目录之外：{raw}")

    parts = set(target.relative_to(ROOT).parts)
    if parts & BLOCKED_NAMES:
        raise PermissionError(f"拒绝访问敏感文件：{raw}")
    if target.suffix.lower() in BLOCKED_SUFFIXES:
        raise PermissionError(f"拒绝访问敏感文件：{raw}")

    return target


# ══════════════════════════════════════════
# 1. 工具实现
# ══════════════════════════════════════════

def list_files(dir: str = ".") -> str:
    path = safe_path(dir)
    if not path.is_dir():
        raise PermissionError(f"不是目录：{dir}")
    return "\n".join(sorted(os.listdir(path)))


def read_file(path: str, lines: int = 20) -> str:
    target = safe_path(path)
    if not target.is_file():
        raise PermissionError(f"不是文件：{path}")
    lines = max(1, min(int(lines), 500))
    with open(target, encoding="utf-8") as f:
        return "".join(f.readlines()[:lines])


def search_docs(question: str) -> str:
    """查 DeepSeek 官方文档。"""
    found = find_hybrid(question)
    context = "\n\n---\n\n".join(found)
    return chat([
        {"role": "system", "content":
            "只根据资料回答。先摘录原文，再给结论。资料里没有就说「资料里没有」。"},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ])


# 代码索引只在第一次调用时构建，之后复用
_code_state = {}


def _code_index(source_path: str = "sample.c"):
    if "cards" not in _code_state:
        cache = load_cache()
        cards = build_cards(source_path, cache, use_summary=True)
        save_cache(cache)
        _code_state["cards"] = cards
        _code_state["bm25"] = build_index(cards)
    return _code_state["cards"], _code_state["bm25"]


def search_code(question: str) -> str:
    """在项目 C 源码里找相关函数。"""
    cards, bm25 = _code_index()
    found = search_cards(question, cards, bm25, 3)
    header = f"在 {len(cards)} 个函数中检索，命中 {len(found)} 个（已按相关度排序，前面的最相关）。"
    body = "\n\n---\n\n".join(c["text"] for c, _ in found)
    return header + "\n\n" + body


TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "search_docs": search_docs,
    "search_code": search_code,
}


# ══════════════════════════════════════════
# 2. 工具说明书
# ══════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出某个目录下的文件名，只能访问项目目录内的路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {"type": "string", "description": "相对路径，例如 data"}
                },
                "required": ["dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文本文件的前若干行，只能访问项目目录内的路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，例如 data/xxx.md"},
                    "lines": {"type": "integer", "description": "读多少行，1-500，默认 20"},
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
                "查 DeepSeek 官方文档。用于：模型名称、模型价格、并发限制、"
                "错误码含义、token 计算、API 调用方式这类问题。"
                "问代码怎么写、哪个函数做了什么，不要用它。"
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
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "在项目的 C 源代码里检索函数。用于：某个功能是由哪个函数实现的、"
                "哪个函数分配了内存、谁处理了空指针、函数在第几行这类问题。"
                "问 DeepSeek 模型的价格或接口参数，不要用它。"
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
                if name not in TOOL_MAP:
                    raise ValueError(f"没有这个工具：{name}")
                result = TOOL_MAP[name](**args)
                print(f"           → 成功，返回 {len(str(result))} 字")
            except Exception as e:
                result = f"工具执行失败：{type(e).__name__}: {e}"
                print(f"           → 被拦下：{e}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "达到最大步数，强制结束"


if __name__ == "__main__":
    questions = [
        "sample.c 里哪个函数分配了内存？",              # → search_code
        "deepseek-flash 的并发限制是多少？",            # → search_docs
        "data 目录下有几个文件？",                      # → list_files
        "你好，你能做什么？",                            # → 不调工具
    ]

    for q in questions:
        print(f"\n{'=' * 56}\n问：{q}\n{'=' * 56}")
        print("答：", run_agent(q))
