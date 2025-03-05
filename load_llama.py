import math
import transformers
import torch
import time

ts = time.time()
lt = time.localtime()

print(f"Loading Model {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")

device = "cpu"

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

model_id = "TinyLlama/TinyLlama_v1.1"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16,
                  "low_cpu_mem_usage": True,
                  },
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "What be the best way to find buried treasure?"},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

lt = time.localtime()
print(f"Model loaded in {math.floor((time.time() - ts)/60):02d}:{round((time.time() - ts)%60):02d} seconds at {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")

ts = time.time()
print(f"Generating response {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])

lt = time.localtime()
print(f"Time taken {math.floor((time.time() - ts)/60):02d}:{round((time.time() - ts)%60):02d} seconds at {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}")