# EleutherAI/pythia-410m-deduped
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="20B",
cache_dir="/dataset/finemath/finemath-llama3b/20B",
)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="20B",
cache_dir="/dataset/finemath/finemath-llama3b/20B",
)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="30B",
cache_dir="/dataset/finemath/finemath-llama3b/30B",
)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="30B",
cache_dir="/dataset/finemath/finemath-llama3b/30B",
)


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="40B",
cache_dir="/dataset/finemath/finemath-llama3b/40B",
)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="40B",
cache_dir="/dataset/finemath/finemath-llama3b/40B",
)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="50B",
cache_dir="/dataset/finemath/finemath-llama3b/50B",
)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/finemath-ablation-finemath-4plus",
revision="50B",
cache_dir="/dataset/finemath/finemath-llama3b/50B",
)