# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

set -evx
# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100"
# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-200"
# MODEL="/home/hossamamer/TTC_checkpoints/TTC-checkpoints/tinyllama-math-code-checkpoint-300"
# MODEL="/home/hossamamer/TTC_checkpoints/TTC-checkpoints/tinyllama-math-code-checkpoint-200"
# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-sft-prm800/from-checkpoint-5000/checkpoint-240/"
# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-sft-prm800/from-checkpoint-31908/checkpoint-240/"

# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-sft-prm800/from-checkpoint-31908/checkpoint-240/"

# MODEL="/home/hossamamer/TTC_checkpoints/tinyllama-sft-prm800/tinyllama-orig-step-50k-105b/"
# MODEL="/home/hossamamer/TTC_checkpoints/tinyllama-sft-prm800/tinyllama-orig-step-240k-503b/"
# MODEL="/home/hossamamer/TTC_workspace/evaluate_math_baseline/pythia-70m-deduped/step3000/models--EleutherAI--pythia-70m-deduped/snapshots/1a4f69ed960a00ecbdae629d21f14d36961285c2"

# MODEL="/home/hossamamer/TTC_workspace/evaluate_math_baseline/pythia-1b-deduped/step143000/models--EleutherAI--pythia-1b-deduped/snapshots/9f638c32a09e234bce2a2da4d37eb08211b816cb/"
# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-sft-prm800/from-checkpoint-10000/"
# MODEL="/home/hossamamer/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5"
# MODEL="meta-llama/Llama-3.2-1B-Instruct"

# MODEL="/data00/maryam/saved_models/tinyllama-sft-prm800/from-checkpoint-5000"
# MODEL="/data00/maryam/saved_models/tinyllama-sft-prm800/from-checkpoint-31908"
# MODEL="/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_baseline_enlarged/checkpoint-834/"

MODEL_PATHS=(
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-87k-SFT-enlarged/checkpoint-834/"
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-92k-SFT-enlarged/checkpoint-834/"
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-97k-SFT-enlarged/checkpoint-834/"
)



# MODEL="/dataset/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8/"

MODEL="/home/hossamamer/TTC_workspace/evaluate_math_baseline/pythia-70m-deduped/step3000/models--EleutherAI--pythia-70m-deduped/snapshots/1a4f69ed960a00ecbdae629d21f14d36961285c2/"

MODEL="/dataset/pythia_models/saved_models/pythia-sft-prm800/70m/from-checkpoint-143000/checkpoint-144"


# MODEL_PATHS=(
#     "/dataset/pythia_models/pythia-410m-deduped/step143000/models--EleutherAI--pythia-410m-deduped/snapshots/c0b6bef7dd1ec11d3baa07ee955de98a414dd464/"
#     "/dataset/pythia_models/saved_models/pythia-sft-prm800/410m/from-checkpoint-143000/"
#     "/dataset/pythia_models/saved_models/pythia-sft-prm800/410m/from-checkpoint-80000/"
#     "/dataset/pythia_models/saved_models/pythia-sft-prm800/410m/from-checkpoint-40000/"
# )

# MODEL_PATHS=(
#     "/dataset/finemath/finemath-llama3b/30B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/b60bc20540d30bc69efb0253a9ea1b4a77ac2054/"
#     "/dataset/finemath/finemath-llama3b/40B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/a5327c94c99a795d0a48089253b8f9356ceed281/"
#     "/dataset/finemath/finemath-llama3b/50B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/49c2b41df57e3e65368f7e2ccdcd50ec3fe88ba8/"
# )

MODEL_PATHS=(
    "/dataset/finemath/finemath-llama3b/120B/models--HuggingFaceTB--finemath-ablation-4plus-160B/snapshots/9ad7077a5473c2ca83d1bce14728660a0f618c34/"
    "/dataset/finemath/finemath-llama3b/80B/models--HuggingFaceTB--finemath-ablation-4plus-160B/snapshots/1902d2e4afb3e01dfdc759c22348ae1884d04543/"
    "/dataset/finemath/finemath-llama3b/160B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/938366e8cae790af6f01aa67cb525a2c14f65561/"
)

# 3 mins per batch of i
for MODEL in "${MODEL_PATHS[@]}"; do
    echo "Processing model at: $MODEL"
    RECIPE=recipes/TinyLlama_v1.1_math_code/dvts.yaml
    for ((i=0; i<500; i+=50)); do
        time python scripts/test_time_compute.py $RECIPE \
        --seed=1 --search_batch_size=25 --prm_batch_size=1 \
        --dataset_start=$i --dataset_end=$((i+50)) \
        --n=1 --model_path=$MODEL \
        --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.45 --beam_width=1
    done
done

