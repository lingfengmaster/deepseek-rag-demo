# DeepSeek 文档问答 RAG

基于 DeepSeek API 的检索增强问答（RAG）系统：把你的技术文档放进去，提问，得到**带原文引用**的答案。

**评测结果：9 道测试题 — BM25 9/9，向量检索 8/9，混合检索（RRF）9/9**

## 快速开始

```bash
git clone https://github.com/lingfengmaster/deepseek-rag-demo.git
cd deepseek-rag-demo
pip install rank-bm25 jieba sentence-transformers openai python-dotenv

echo "DEEPSEEK_API_KEY=sk-你的key" > .env
python rag.py

它做什么
把 data/ 里的 Markdown 切成片段，表格按行切、每行带表头
用 BM25（关键词）/ 向量（语义）/ RRF 混合三种方式，召回最相关的 3 段
交给 DeepSeek 生成答案，强制「先引用原文，再下结论」
核心设计
1. 表格必须按行切
朴素的按空行切分会把整张 22 行的价格表压成一个 4000 字的片段。BM25 有长度归一化惩罚，长文档得分反而更低，导致答案明明在资料里却检索不到。

解法：表格按行切成独立片段，每行拼上去重后的表头，并去掉对齐用的多余空格。

2. 三种检索并存，用数据说话
检索方式	得分	特点
BM25	9/9	强在精确匹配（型号、数字、条款）
向量（bge-small-zh-v1.5）	8/9	强在同义表达，但在多行高度相似的表格上会随机召回
RRF 混合	9/9	本语料无额外收益，说明术语规范时 BM25 已足够
评测集：6 道事实题 + 3 道语义改写题，答案人工核对。

3. Prompt 先证据后结论
系统提示强制「先摘录原文，再给结论」，解决了模型先否认后补充的自相矛盾问题（输出 token 从 3073 降到 687）。

实验结果
详见 experiment.md：6 轮迭代的完整记录，包括失败版本和原因分析。

版本	改动	结果
v1	按空行切分 + BM25	2/4
v2	表格按行切 + 表头拼接	6/6
v3	Prompt 加「先摘录后结论」	消除自相矛盾
v4-v6	加语义题 + 向量 + RRF 对比	9/9 / 8/9 / 9/9
技术栈
Python · BM25 (rank_bm25) · jieba · sentence-transformers (bge-small-zh-v1.5) · DeepSeek API · NumPy

目录结构
代码块


rag.py       主程序：切分 → 检索 → 生成
llm.py       LLM 调用封装（重试 / 超时 / token 统计）
data/        语料（5 篇 DeepSeek 官方文档）
experiment.md  实验记录
后续计划
 FastAPI 接口 + Streamlit 界面
 Docker 一键部署
 支持 PDF / Word 入库
