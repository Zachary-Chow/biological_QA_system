from sentence_transformers import SentenceTransformer
# gte_model = model = SentenceTransformer('gte-Qwen2-7B-instruct', cache_folder=r"../model")
# gte_model = SentenceTransformer("../model/gte-Qwen2-7B-instruct", trust_remote_code=True)  # 粗排encoder
# gte_model = SentenceTransformer("../model/all-mpnet-base-v2", trust_remote_code=True)  # 粗排encoder
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", cache_folder=r"../model")
# model = SentenceTransformer("BAAI/bge-m3", cache_folder=r"../model", trust_remote_code=True)