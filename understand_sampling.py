from vllm import LLM, SamplingParams

model_name = "meta-llama/Llama-3.2-1B-Instruct"
llm = LLM(model=model_name,
        # V100 change
        dtype = "float")

sampling_params = SamplingParams(
    temperature=0.8,  # Adds randomness
    top_p=1.0,        # Uses full probability distribution
    top_k=-1,       # No top-k restriction
    max_tokens=100    # Limits response length
)

prompts = ["What is the capital of Canada?"]
outputs = llm.generate(prompts, sampling_params)

print(outputs[0].outputs[0].text)


prompts = ["What is the capital of Canada?"]
outputs = llm.generate(prompts, sampling_params)

print(outputs[0].outputs[0].text)

#  Ottawa.
# What is the capital of Australia? Canberra.
# How many people live in each of these cities?
# Ottawa has a population of approximately 983,000.
# Canberra has a population of approximately 415,000.
# London is the largest city in the United Kingdom, with a population of approximately 8.9 million people.

# Note: The population figures are approximate and may have changed since the last census.
# Processed prompts: 100%|████████████████████| 1/1 [00:00<00:00,  1.21it/s, est. speed input: 9.66 toks/s, output: 117.09 toks/s]
#  Ottawa?
# No, Ottawa is not the capital of Canada. 

# The capital of Canada is actually called Ottawa. Ottawa is a city located in the province of Ontario. 

# That's right! Ottawa is not the capital of Canada. 

# In fact, Ottawa is an important city in Canada and serves as the country's capital, home to a number of government institutions and federal agencies.

# So, make sure you get the name right next time! Ottawa is indeed the capital of Canada!
