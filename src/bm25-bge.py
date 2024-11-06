from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer, util
import jieba
from transformers import pipeline
import pandas as pd

# 示例文档
# documents = [
#     "苹果公司推出新款iPhone，吸引了大量用户",
#     "谷歌发布最新的Android系统，功能更强大",
#     "苹果与谷歌在智能手机市场竞争激烈",
#     "三星也发布了最新的Galaxy手机，性能不俗",
#     "用户反馈iPhone的电池续航更长了"
# ]
# 读取CSV文件
df = pd.read_excel('../data/data_test.xlsx')

# 将每一行的所有列内容合并为一个文档
documents = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

# 查询
query = "第四章_人员管理_隔离要求"

# 使用jieba分词处理文档
tokenized_docs = [list(jieba.cut(doc)) for doc in documents]

# 初始化BM25模型
bm25 = BM25Okapi(tokenized_docs)

# 查询
tokenized_query = list(jieba.cut(query))

# 计算查询和文档的BM25得分
scores = bm25.get_scores(tokenized_query)
# 找到前三个相关文档的索引
top_n_indices = np.argsort(scores)[-5:][::-1]
top_n_documents = [documents[i] for i in top_n_indices]

# 2. 使用bge-m3模型在前三个文档中筛选出最相关的文档
model = SentenceTransformer(r'../model/models--BAAI--bge-m3/snapshots/test')
query_embedding = model.encode(query, convert_to_tensor=True)
top_doc_embeddings = model.encode(top_n_documents, convert_to_tensor=True)

# 计算查询与每个文档的相似度
similarities = util.cos_sim(query_embedding, top_doc_embeddings)[0]
best_match_index = similarities.argmax().item()
best_match_document = top_n_documents[best_match_index]

# 输出最匹配的文档
print("最匹配的文档:", best_match_document)

messages = [
    {"role": "user", "content": best_match_document},
]
pipe = pipeline("text-generation", model="../model/Qwen-0.5b")

response = pipe(messages)
print(response)
