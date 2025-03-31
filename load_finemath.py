# EleutherAI/pythia-410m-deduped
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="10B",
cache_dir="/dataset/finemath/finemath-llama3b/10B",
)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="10B",
cache_dir="/dataset/finemath/finemath-llama3b/10B",
)

