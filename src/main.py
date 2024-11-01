import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# 数据准备
def prepare_data():
    # 加载数据集，这里以简单的例子为例
    dataset = load_dataset('squad', split='train[:10%]')
    contexts = [example['context'] for example in dataset]
    return contexts


# 创建BM25索引
def create_bm25_index(contexts):
    vectorizer = TfidfVectorizer(use_idf=False)
    X = vectorizer.fit_transform(contexts)
    index = faiss.IndexFlatIP(X.shape[1])
    faiss.normalize_L2(X.toarray())
    index.add(X.toarray())
    return index, vectorizer


# 创建SentenceTransformer索引
def create_sentence_transformer_index(contexts, gte_model):
    model = gte_model
    embeddings = model.encode(contexts, convert_to_tensor=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings.cpu().numpy())
    index.add(embeddings.cpu().numpy())
    return index, model


# 检索函数
def retrieve_documents(query, bm25_index, st_index, bm25_vectorizer, st_model, top_k=5):
    # BM25检索
    query_vec = bm25_vectorizer.transform([query])
    faiss.normalize_L2(query_vec.toarray())
    _, bm25_indices = bm25_index.search(query_vec.toarray(), top_k)

    # SentenceTransformer检索
    query_emb = st_model.encode([query], convert_to_tensor=True)
    faiss.normalize_L2(query_emb.cpu().numpy())
    _, st_indices = st_index.search(query_emb.cpu().numpy(), top_k)

    # 融合结果
    fused_indices = list(set(bm25_indices[0]) | set(st_indices[0]))
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
    new_prompt = f"Context: {' '.join(documents)}\nQuestion: {question}"
    answer = qa_pipeline(new_prompt)
    return answer


# 主函数
def main():
    # 加载模型
    gte_model = SentenceTransformer("../model/gte-Qwen2-7B-instruct", trust_remote_code=True)  # 粗排encoder
    bge_model = SentenceTransformer('../models/bge-m3')  # 精排encoder

    # 加载千问7B模型
    model_name = "../model/qwen-7b"
    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForCausalLM.from_pretrained(model_name)
    # 创建生成管道
    qa_pipeline = pipeline("text-generation", model=qa_model, tokenizer=qa_tokenizer)

    # 准备数据
    contexts = prepare_data()

    # 创建索引
    bm25_index, bm25_vectorizer = create_bm25_index(contexts)
    st_index, st_model = create_sentence_transformer_index(contexts, gte_model)

    # 示例问题
    question = "烘干房后人员_车辆烘干_种猪场生物安全管理"

    # 检索文档
    retrieved_docs = retrieve_documents(question, bm25_index, st_index, bm25_vectorizer, st_model)

    # 精排
    ranked_docs = rerank_documents(retrieved_docs, question, bge_model)

    # 生成答案
    answer = generate_answer(question, ranked_docs, qa_pipeline)
    print("Answer:", answer['answer'])


if __name__ == "__main__":
    main()
