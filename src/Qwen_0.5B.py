# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you? you speak chinese and only response one sentence"},
]
pipe = pipeline("text-generation", model="../model/Qwen-0.5b")

response = pipe(messages)
print(response)
