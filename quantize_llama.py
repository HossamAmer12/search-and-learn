import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
# from bitsandbytes import quantize

# Model name (replace with correct Hugging Face repo if needed)
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Choose quantization type: '4bit' or '8bit'
QUANTIZATION_TYPE = "8bit"  # Change to "8bit" if needed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with quantization
if QUANTIZATION_TYPE == "4bit":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
elif QUANTIZATION_TYPE == "8bit":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )
else:
    raise ValueError("Invalid quantization type. Use '4bit' or '8bit'.")

# Move to V100 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Test inference
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
    
# Decode and print result
print(tokenizer.decode(output[0], skip_special_tokens=True))
