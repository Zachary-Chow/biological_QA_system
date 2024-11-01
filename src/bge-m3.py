from sentence_transformers import SentenceTransformer

# import os
# # 设置环境变量以强制离线模式
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

# bge_model = SentenceTransformer('../model/models--BAAI--bge-m3')  # 精排encoder
bge_model = SentenceTransformer('BAAI/bge-m3', cache_folder=r'../model/models--BAAI--bge-m3')
sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = bge_model.encode(sentences)

similarities = bge_model.similarity(embeddings, embeddings)
print(similarities.shape)
# [4, 4]
