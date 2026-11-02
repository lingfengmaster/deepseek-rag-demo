import os, time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

def chat(messages, model="deepseek-flash", temperature=0.7, retries=3):
    """带重试、超时、token 统计的对话封装。"""
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=30,
            )
            u = r.usage
            print(f"[tokens] in={u.prompt_tokens} out={u.completion_tokens}")
            return r.choices[0].message.content
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** i          # 1s, 2s, 4s 指数退避
            print(f"[retry {i+1}] {type(e).__name__}: {e}, {wait}s 后重试")
            time.sleep(wait)

# model="deepseek-flash"      # 快、便宜，日常开发用这个
# model="deepseek-v4-pro"   # 难任务再用，贵

def chat_stream(messages, model="deepseek-flash", show=True):
    """边打边显示，同时把完整回答返回。"""
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)          # 攒起来
            if show:
                print(delta, end="", flush=True)   # 同时显示
    if show:
        print()
    return "".join(parts)                # 关键：返回全文