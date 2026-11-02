"""RAG 最小可用版：同时支持 BM25 和向量两种检索，方便对比。"""

import glob
import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from llm import chat

# ══════════════════════════════════════════
# 1. 切分
# ══════════════════════════════════════════

def split_md(text):
    """按段落切；表格按行切，每行带上去重后的表头，单元格压扁。"""
    chunks, buf, table = [], [], []

    def flush_buf():
        if buf:
            p = "\n".join(buf).strip()
            if len(p) > 30:
                chunks.append(p)
            buf.clear()

    def head_of(row):
        cells = [c.strip() for c in row.strip("|").split("|")]
        return " | ".join(dict.fromkeys(c for c in cells if c))

    def compact(row):
        cells = [c.strip() for c in row.strip("|").split("|")]
        return " | ".join(c for c in cells if c)

    def flush_table():
        if not table:
            return
        head = head_of(table[0])
        for row in table[1:]:
            if re.fullmatch(r"\|[\s:\-|]+\|", row):
                continue
            chunks.append(f"{head}\n{compact(row)}")
        table.clear()

    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            flush_buf()
            table.append(s)
        elif s:
            flush_table()
            buf.append(s)
        else:
            flush_table()
            flush_buf()
    flush_table()
    flush_buf()
    return chunks


chunks = []
for _f in glob.glob("data/*.md"):
    chunks += split_md(open(_f, encoding="utf-8").read())
print(f"共 {len(chunks)} 张纸条")

# ══════════════════════════════════════════
# 2. 检索器 A：BM25（按关键词）
# ══════════════════════════════════════════

corpus = [list(jieba.cut(c)) for c in chunks]
bm25 = BM25Okapi(corpus)


def find_bm25(question, k=3):
    scores = bm25.get_scores(list(jieba.cut(question)))
    order = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in order]

# ══════════════════════════════════════════
# 3. 检索器 B：向量（按意思）
# ══════════════════════════════════════════

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vectors = model.encode(chunks, normalize_embeddings=True, batch_size=32)


def find_vector(question, k=3):
    q = "为这个句子生成表示以用于检索相关文章：" + question
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = vectors @ qv
    order = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in order]


def _bm25_ranking(question):
    scores = bm25.get_scores(list(jieba.cut(question)))
    return list(np.argsort(scores)[::-1])


def _vector_ranking(question):
    q = "为这个句子生成表示以用于检索相关文章：" + question
    qv = model.encode([q], normalize_embeddings=True)[0]
    return list(np.argsort(vectors @ qv)[::-1])


def find_hybrid(question, k=3):
    """两路各出一份排名，按名次投票（RRF）。"""
    scores = {}
    for ranking in (_bm25_ranking(question), _vector_ranking(question)):
        for rank, idx in enumerate(ranking[:20]):
            scores[idx] = scores.get(idx, 0) + 1 / (60 + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [chunks[i] for i in best]
# ══════════════════════════════════════════
# 4. 生成
# ══════════════════════════════════════════

def ask(question, retriever, debug=False):
    found = retriever(question)

    if debug:
        print(f"\n{'=' * 50}\n问题：{question}")
        for i, c in enumerate(found, 1):
            print(f"[{i}] {len(c)} 字：{c[:100]}")
        print("=" * 50)

    context = "\n\n---\n\n".join(found)
    return chat([
        {"role": "system", "content":
            "你是资料问答助手，遵守三条规则：\n"
            "1. 只根据下面的资料回答，不要使用你自己的知识。\n"
            "2. 先摘录资料里的相关原文，再给出结论。\n"
            "3. 资料里确实没有的，直接说「资料里没有」，不要猜。"},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ])

# ══════════════════════════════════════════
# 5. 测试集（6 题，括号里是标准答案）
# ══════════════════════════════════════════

QUESTIONS = [
    "1 个中文字符大约等于多少 token？",                # 0.6
    "400 错误码是什么原因？",                         # 请求体格式错误
    "deepseek-flash 的并发限制是多少？",              # 2500
    "deepseek-v4-pro 百万 tokens 输出的价格是多少？",  # 高峰 27.0 / 空闲 13.5
    "deepseek-flash 缓存命中的输入价格是多少？",       # 空闲 0.02 / 高峰 0.04
    "DeepSeek 支持微调自己的模型吗？",                 # 资料里没有

    # ↓ 新增三题：故意不用资料里的原词，检验"换个说法"能不能找到
    "请求发得太频繁会被限制吗？",                      # 会，超过并发限度收到 HTTP 429
    "账户余额不够了会怎样？",                          # 402 余额不足
    "费用到底是怎么算的？",                            # 模型调用费用 = token 消耗量 × 单价
]

if __name__ == "__main__":
    for name, fn in [("BM25（关键词）", find_bm25), ("向量（语义）", find_vector),("混合（RRF）", find_hybrid)]:
        print(f"\n{'#' * 56}\n# {name}\n{'#' * 56}")
        for i, q in enumerate(QUESTIONS, 1):
            ans = ask(q, fn).strip().replace("\n", " ")
            # 答案结构是「资料原文：… 结论：…」，只留结论部分
            if "结论" in ans:
                ans = ans[ans.index("结论"):]
            print(f"\n[{i}] {q}")
            print(f"    {ans}")