import json
import os
from tqdm import tqdm 

tenB = "data/dataset/finemath/finemath-llama3b/10B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/f3be85d2df204cf454cfd06657b7b0c788ceedb1/output/dvts_completions.jsonl"
# twentyB = "/home/hossamamer/TTC_workspace/search-and-learn/data/dataset/finemath/finemath-llama3b/20B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/4ccc2949da259213552d6257a89c9448a666437e/output/dvts_completions.jsonl"
thirtyB = "/home/hossamamer/TTC_workspace/search-and-learn/data/dataset/finemath/finemath-llama3b/30B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/b60bc20540d30bc69efb0253a9ea1b4a77ac2054/output/dvts_completions.jsonl"
fourtyB = "/home/hossamamer/TTC_workspace/search-and-learn/data/dataset/finemath/finemath-llama3b/40B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/a5327c94c99a795d0a48089253b8f9356ceed281/output/dvts_completions.jsonl"
fiftyB = "/home/hossamamer/TTC_workspace/search-and-learn/data/dataset/finemath/finemath-llama3b/50B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/49c2b41df57e3e65368f7e2ccdcd50ec3fe88ba8/output/dvts_completions.jsonl"

# List of input filenames
# filenames = ["10B.json", "30B.json", "40B.json", "50B.json"]
filenames = [tenB, thirtyB, fourtyB, fiftyB]

onames = ["10B.json", "30B.json", "40B.json", "50B.json"]


# Load data from each file
# Properly load each line as an individual JSON object
all_data = {}
for fname in filenames:
    with open(fname) as f:
        lines = f.readlines()
        all_data[fname] = [json.loads(line) for line in lines]

# all_data = {fname: json.load(open(fname)) for fname in filenames}


# Helper to extract problem keys
def get_problem_keys(data):
    return set(p["problem"] for p in data if "problem" in p)

# Function to extract top-N problem keys
def get_top_problem_keys(problem_list, n):
    return set(p["problem"] for p in problem_list[:n] if "problem" in p)

# Process each file
for i, target_file in (enumerate(filenames)):
    # Get current file data
    current_data = all_data[target_file]

    # Get intersection of problem keys from other files
    other_files = [f for f in filenames if f != target_file]
    other_keys = [get_problem_keys(all_data[f]) for f in other_files]
    intersect_keys = set.intersection(*other_keys)

    # From the current file, extract matching problems
    result = [p for p in current_data if p.get("problem") in intersect_keys]

    # Write to output
    out_file = onames[i]
    out_file = os.path.join("data/finemath_clean/", out_file)
    with open(out_file, "w") as out:
        for entry in result:
            out.write(json.dumps(entry) + "\n")

    print("Length: ", len(result))
    # with open(out_file, "w") as out:
    #     json.dump(result, out, indent=2)

    print(f"Created: {out_file}")
