# EleutherAI/pythia-410m-deduped
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# sampled_ckpt_list = [1000, 20000, 40000, 80000, 100000, 120000, 143000]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped",
revision="step40000",
cache_dir="/dataset/pythia_models//pythia-410m-deduped/step40000",
)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped",
revision="step40000",
cache_dir="/dataset/pythia_models/pythia-410m-deduped/step40000",
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped",
revision="step143000",
cache_dir="/dataset/pythia_models//pythia-410m-deduped/step143000",
)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped",
revision="step143000",
cache_dir="/dataset/pythia_models/pythia-410m-deduped/step143000",
)
