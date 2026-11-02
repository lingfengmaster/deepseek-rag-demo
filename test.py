import numpy as np
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("BAAI/bge-small-zh-v1.5")
texts = ["价格是多少", "费用多少", "今天天气", "模型贵不贵"]
v = m.encode(texts, normalize_embeddings=True)

print(np.round(v @ v.T, 3))