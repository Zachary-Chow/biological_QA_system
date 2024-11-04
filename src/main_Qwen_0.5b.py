import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity

import faiss


# 数据准备
def prepare_data():
    # 指定本地 CSV 文件路径
    csv_file_path = r"../data/data.csv"
    # 加载 CSV 文件
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    # 检查 DataFrame 的前几条记录
    # print(df.head())

    # 提取每一行的正文作为 chunk
    chunks = df['正文'].dropna().tolist()  # dropna() 用于去除空值

    return chunks


# 创建BM25索引
# 创建 BM25 检索器
def create_bm25_index(contexts):
    # 将文本数据转换为词袋表示
    texts = [doc.split() for doc in contexts]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 创建 TF-IDF 模型
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # 创建 BM25 模型
    bm25 = SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

    return bm25, dictionary, tfidf


# BM25 检索
def bm25_search(bm25, dictionary, tfidf, query, top_k):
    query_bow = dictionary.doc2bow(query.split())
    query_tfidf = tfidf[query_bow]
    sims = bm25[query_tfidf]
    sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])[:top_k]
    bm25_indices = [i for i, _ in sorted_sims]
    return bm25_indices


# 创建SentenceTransformer索引
def create_sentence_transformer_index(contexts, gte_model):
    model = gte_model
    embeddings = model.encode(contexts, convert_to_tensor=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings.cpu().numpy())
    index.add(embeddings.cpu().numpy())
    return index, model


# 检索函数
def retrieve_documents(question, bm25, dictionary, tfidf, st_index, st_model, top_k=5,
                       contexts=None) -> list:
    """

    :param query:
    :param bm25_index:
    :param st_index:
    :param bm25_vectorizer:
    :param st_model:
    :param top_k:
    :param contexts:
    :return:list 返回粗排后的知识库list
    """
    # BM25 检索
    bm25_indices = bm25_search(bm25, dictionary, tfidf, question, top_k)

    # SentenceTransformer 检索（假设已经实现）
    # query_emb = st_model.encode([query], convert_to_tensor=True)
    # faiss.normalize_L2(query_emb.cpu().numpy())
    # _, st_indices = st_index.search(query_emb.cpu().numpy(), top_k)

    # 融合结果
    fused_indices = list(set(bm25_indices))  # 你可以在这里融合 SentenceTransformer 的结果
    return [contexts[i] for i in fused_indices]


# 精排
def rerank_documents(documents, question, bge_model):
    inputs = [f"{question} [SEP] {doc}" for doc in documents]
    embeddings = bge_model.encode(inputs, convert_to_tensor=True)
    scores = torch.sum(embeddings * bge_model.encode([question] * len(documents), convert_to_tensor=True), dim=1)
    sorted_scores, indices = torch.sort(scores, descending=True)
    return [documents[i] for i in indices]


# 生成答案
def generate_answer(question, documents, qa_pipeline):
    new_prompt = f"reference: {' '.join(documents)}\nQuestion: {question}"
    print("new_prompt:", new_prompt)
    answer = qa_pipeline(new_prompt)
    return answer


# 主函数
def main():
    # 加载模型
    # gte_model = SentenceTransformer("../model/gte-Qwen2-7B-instruct", trust_remote_code=True)  # 粗排encoder
    bge_model = SentenceTransformer(
        '../model/models--BAAI--bge-m3/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181')  # 精排encoder

    # 加载千问3B模型
    model_name = "../model/Qwen-3b"
    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForCausalLM.from_pretrained(model_name)
    # 创建生成管道
    qa_pipeline = pipeline("text-generation", model=qa_model, tokenizer=qa_tokenizer)
    # 修改max_length
    qa_pipeline.model.generation_config.max_length = 2000
    # qa_pipeline = pipeline("text-generation", model="../model/Qwen-0.5b")

    # 准备数据
    contexts = prepare_data()

    # 创建 BM25 索引
    bm25, dictionary, tfidf = create_bm25_index(contexts)

    # st_index, st_model = create_sentence_transformer_index(contexts, gte_model)

    # 示例问题
    question = "烘干房后人员_车辆烘干_种猪场生物安全管理"

    # 检索文档
    st_index = None
    st_model = None
    retrieved_docs = retrieve_documents(question, bm25, dictionary, tfidf, st_index, st_model, top_k=5,
                                        contexts=contexts)

    # 精排
    ranked_docs = rerank_documents(retrieved_docs, question, bge_model)

    # 生成答案
    answer = generate_answer(question, ranked_docs, qa_pipeline)
    print("Answer:", answer[0]['generated_text'])


if __name__ == "__main__":
    main()
