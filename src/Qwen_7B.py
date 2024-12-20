# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载千问7B模型
model_name = f"../model/qwen-7b"
qa_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
qa_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# 创建生成管道
qa_pipeline = pipeline("text-generation", model=qa_model, tokenizer=qa_tokenizer)
