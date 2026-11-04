"""代码库问答：解析 C 函数 → 生成中文摘要 → 检索 → 回答。

核心对比：
    不加摘要：中文问题搜不到英文代码
    加了摘要：中文问题能命中

用法：
    python code_rag.py sample.c
"""

import json
import os
import sys

import jieba
from rank_bm25 import BM25Okapi

from llm import chat
from parse_c import parse_file

CACHE_FILE = ".cache/func_summaries.json"

# 改了摘要 prompt 就改这个版本号 —— 否则会一直命中旧缓存
PROMPT_VERSION = "v2"


# ══════════════════════════════════════════
# 1. 摘要缓存
# ══════════════════════════════════════════

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def summarize(func, cache):
    """让模型用一句中文说明这个函数做什么，结果缓存起来。"""
    key = f"{PROMPT_VERSION}::{func['file']}::{func['name']}::{func['start_line']}"
    if key in cache:
        return cache[key]

    text = chat([
        {"role": "system", "content":
            "你是 C 代码分析助手。用一句中文说明这个函数做什么，不超过 50 字。"
            "必须提到两点：① 主要操作（内存分配、字符串拼接、数值比较、打印输出等）；"
            "② 是否处理空指针或失败情况。"
            "只描述功能，不要复述代码。"},
        {"role": "user", "content": func["code"]},
    ])
    cache[key] = text.strip()
    return cache[key]


# ══════════════════════════════════════════
# 2. 建卡片
# ══════════════════════════════════════════

def build_cards(path, cache, use_summary=True):
    """把每个函数变成一张可检索的卡片。"""
    cards = []
    for func in parse_file(path):
        head = f"文件 {func['file']} 函数 {func['name']}（第 {func['start_line']}-{func['end_line']} 行）"

        if use_summary:
            summary = summarize(func, cache)
            text = f"{head}\n功能：{summary}\n签名：{func['signature']}\n源码：\n{func['code']}"
        else:
            summary = "（无）"
            text = f"{head}\n签名：{func['signature']}\n源码：\n{func['code']}"

        cards.append({
            "name": func["name"],
            "lines": f"{func['start_line']}-{func['end_line']}",
            "summary": summary,
            "code": func["code"],
            "text": text,
        })
    return cards


# ══════════════════════════════════════════
# 3. 检索
# ══════════════════════════════════════════

def build_index(cards):
    corpus = [list(jieba.cut(c["text"])) for c in cards]
    return BM25Okapi(corpus)


def search(question, cards, bm25, k=3):
    scores = bm25.get_scores(list(jieba.cut(question)))
    order = sorted(range(len(cards)), key=lambda i: scores[i], reverse=True)[:k]
    return [(cards[i], scores[i]) for i in order]


def ask(question, cards, bm25, k=3):
    found = search(question, cards, bm25, k)
    context = "\n\n---\n\n".join(c["text"] for c, _ in found)
    return chat([
        {"role": "system", "content":
            "你是 C 代码助手。根据给出的函数回答，先说明是哪个函数（带行号），再解释。"
            "资料里没有就说「代码库里没有」。"},
        {"role": "user", "content": f"函数资料：\n{context}\n\n问题：{question}"},
    ])


# ══════════════════════════════════════════
# 4. 测试集：每道题都带标准答案
# ══════════════════════════════════════════

QUESTIONS = [
    ("哪个函数把数值限制在上下限之间？", "clamp"),
    ("哪个函数会打印用户信息？", "print_user"),
    ("哪个函数在内存分配失败时会返回空？", "make_name"),
]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.c"
    cache = load_cache()

    cards_with = build_cards(path, cache, use_summary=True)
    save_cache(cache)

    cards_without = build_cards(path, {}, use_summary=False)

    print(f"已生成 {len(cards_with)} 张函数卡片\n")
    for c in cards_with:
        print(f"  {c['name']:<12} 第 {c['lines']:<8} {c['summary']}")
    print()

    idx_with = build_index(cards_with)
    idx_without = build_index(cards_without)

    print("=" * 78)
    print("对比实验：同一个问题，两种建索引方式，各自命中的第一个函数")
    print("=" * 78)
    print(f"{'问题':<30}{'无摘要':<14}{'有摘要':<14}{'标准答案'}")
    print("-" * 78)

    right_none = right_with = 0
    for q, expected in QUESTIONS:
        top_none = search(q, cards_without, idx_without, 1)[0][0]["name"]
        top_with = search(q, cards_with, idx_with, 1)[0][0]["name"]

        ok_none = "✅" if top_none == expected else "❌"
        ok_with = "✅" if top_with == expected else "❌"
        right_none += top_none == expected
        right_with += top_with == expected

        print(f"{q[:16]:<30}{top_none + ok_none:<14}{top_with + ok_with:<14}{expected}")

    print("-" * 78)
    print(f"{'正确率':<30}{f'{right_none}/{len(QUESTIONS)}':<14}{f'{right_with}/{len(QUESTIONS)}'}")

    print("\n" + "=" * 78)
    print("用「有摘要」的索引回答第三个问题：")
    print("=" * 78)
    print(ask(QUESTIONS[2][0], cards_with, idx_with))
