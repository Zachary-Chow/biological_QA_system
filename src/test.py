from rank_bm25 import BM25Okapi
import numpy as np
import jieba

# 示例文档
documents = [
    "苹果公司推出新款iPhone，吸引了大量用户",
    "谷歌发布最新的Android系统，功能更强大",
    "苹果与谷歌在智能手机市场竞争激烈",
    "三星也发布了最新的Galaxy手机，性能不俗",
    "用户反馈iPhone的电池续航更长了"
]

# 使用jieba分词处理文档
tokenized_docs = [list(jieba.cut(doc)) for doc in documents]

# 初始化BM25模型
bm25 = BM25Okapi(tokenized_docs)

# 查询
query = "苹果 iPhone"
tokenized_query = list(jieba.cut(query))

# 计算查询和文档的BM25得分
scores = bm25.get_scores(tokenized_query)

# 输出BM25得分
print("BM25得分:", scores)

# 找到前三个相关文档的索引
top_n_indices = np.argsort(scores)[-3:][::-1]
top_n_documents = [documents[i] for i in top_n_indices]

# 输出前三个最相关文档
print("前三个最相关文档:", top_n_documents)
