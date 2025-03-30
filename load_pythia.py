# EleutherAI/pythia-70m-deduped
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped",
revision="step143000",
cache_dir="./pythia-1b-deduped/step143000",
)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped",
revision="step143000",
cache_dir="./pythia-1b-deduped/step143000",
)

