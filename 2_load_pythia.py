# EleutherAI/pythia-410m-deduped
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped",
revision="step40000",
cache_dir="/dataset/pythia-410m-deduped//step40000",
)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped",
revision="step40000",
cache_dir="/dataset/pythia-410m-deduped//step40000",
)

