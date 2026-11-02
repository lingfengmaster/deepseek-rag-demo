from llm import chat_stream

text = chat_stream(
    messages=[{"role": "user", "content": "用一句话解释什么是 RAG"}]
)
print("\n--- 一共", len(text), "个字 ---")