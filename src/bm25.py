from rank_bm25 import BM25Okapi

# 示例文档
documents = [
    "苹果公司推出新款iPhone，吸引了大量用户",
    "谷歌发布最新的Android系统，功能更强大",
    "苹果与谷歌在智能手机市场竞争激烈"
]

# 对文档进行分词
tokenized_docs = [doc.split(" ") for doc in documents]

# 初始化BM25模型
bm25 = BM25Okapi(tokenized_docs)

# 查询
query = "苹果 iPhone"
tokenized_query = query.split(" ")

# 计算查询和文档的BM25得分
scores = bm25.get_scores(tokenized_query)

# 找到最匹配的文档
best_match_index = scores.argmax()
best_match_document = documents[best_match_index]

# 输出最匹配的文档
print("最匹配的文档:", best_match_document)
