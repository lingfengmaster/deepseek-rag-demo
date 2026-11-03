"""LLM 调用封装：统一处理重试、超时、token 统计。

对外三个函数：
    chat()        —— 返回字符串（普通对话）
    chat_raw()    —— 返回原始 message 对象（需要用 tool_calls 时）
    chat_stream() —— 流式输出，同时返回完整字符串
"""

import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("没读到 DEEPSEEK_API_KEY，检查 ~/deepseek_test/.env 是否存在、变量名是否一致")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

DEFAULT_MODEL = "deepseek-flash"


def _is_retryable(e: Exception) -> bool:
    """只重试"可能自己好"的错误。

    网络断了、超时、限流（429）、服务端 5xx —— 这些重试有意义。
    参数错（400）、认证失败（401）—— 重试一百次也一样，立刻抛。
    """
    if isinstance(e, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(e, APIStatusError):
        return e.status_code >= 500
    return False


def _create(retries: int = 3, **kwargs):
    """带重试地调一次接口，返回原始 message 对象。"""
    for i in range(retries):
        try:
            resp = client.chat.completions.create(timeout=60, **kwargs)
            u = resp.usage
            print(f"[tokens] in={u.prompt_tokens} out={u.completion_tokens}")
            return resp.choices[0].message
        except Exception as e:
            if not _is_retryable(e) or i == retries - 1:
                raise
            wait = 2 ** i
            print(f"[retry {i + 1}] {type(e).__name__}, {wait}s 后重试")
            time.sleep(wait)


def chat(messages, model=DEFAULT_MODEL, temperature=0.7, retries=3):
    """普通对话，返回字符串。"""
    msg = _create(
        model=model,
        messages=messages,
        temperature=temperature,
        retries=retries,
    )
    return msg.content


def chat_raw(messages, model=DEFAULT_MODEL, tools=None, retries=3):
    """需要工具调用时用，返回原始 message（带 tool_calls）。"""
    kwargs = dict(model=model, messages=messages, retries=retries)
    if tools:
        kwargs["tools"] = tools
    return _create(**kwargs)


def chat_stream(messages, model=DEFAULT_MODEL, show=True):
    """边打边显示，同时把完整回答返回。"""
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            if show:
                print(delta, end="", flush=True)
    if show:
        print()
    return "".join(parts)
