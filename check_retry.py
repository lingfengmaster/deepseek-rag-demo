"""验证 llm.chat 的重试逻辑。

用法（在 ~/deepseek_test 目录下）：
    python check_retry.py

看什么：
    看最后两行结论。总耗时是关键指标。
    - 耗时 ≈ 3 秒  → 400 错误也被重试了（宽泛 except 的副作用）
    - 耗时 < 1 秒  → 只重试可恢复错误，这是正确行为

这个小脚本可以反复用：改完 llm.py 的重试逻辑，再跑一次，
耗时从 3 秒变成 0.3 秒，就是改动生效的证据。
"""

import time
from llm import chat

print("=" * 56)
print("测试：故意传一个不存在的模型名，观察重试行为")
print("=" * 56)

start = time.perf_counter()

try:
    # 这个模型名不存在，服务端一定返回 400
    chat(
        [{"role": "user", "content": "hi"}],
        model="not-exist-model",
    )
    print("\n❌ 居然没报错？说明 model 参数根本没传给接口")

except Exception as e:
    cost = time.perf_counter() - start

    print("\n" + "-" * 56)
    print(f"抛出的异常：{type(e).__name__}")
    print(f"总耗时：{cost:.2f} 秒")
    print("-" * 56)

    if cost >= 2.5:
        print("结论：重试了 3 次（1s + 2s 指数退避）。")
        print("      但这个 400 错误永远不会成功，重试纯属浪费时间。")
        print("      改进：把 except Exception 换成只抓 RETRYABLE。")
    else:
        print("结论：立刻失败，没有无效重试。这是正确行为。")
        print("      可恢复错误（超时/429/5xx）仍会重试，不影响稳定性。")