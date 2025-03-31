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

# MODEL="/dataset/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8/"

MODEL="/home/hossamamer/TTC_workspace/evaluate_math_baseline/pythia-70m-deduped/step3000/models--EleutherAI--pythia-70m-deduped/snapshots/1a4f69ed960a00ecbdae629d21f14d36961285c2/"

MODEL="/dataset/pythia_models/saved_models/pythia-sft-prm800/70m/from-checkpoint-143000/checkpoint-144"


# MODEL_PATHS=(
#     "/dataset/pythia_models/pythia-410m-deduped/step143000/models--EleutherAI--pythia-410m-deduped/snapshots/c0b6bef7dd1ec11d3baa07ee955de98a414dd464/"
#     "/dataset/pythia_models/saved_models/pythia-sft-prm800/410m/from-checkpoint-143000/"
#     "/dataset/pythia_models/saved_models/pythia-sft-prm800/410m/from-checkpoint-80000/"
#     "/dataset/pythia_models/saved_models/pythia-sft-prm800/410m/from-checkpoint-40000/"
# )

MODEL_PATHS=(
    "/dataset/saved_models/pythia-sft-prm800/70m/from-checkpoint-143000/checkpoint-834/"
    "/dataset/saved_models/pythia-sft-prm800/70m/from-checkpoint-80000/checkpoint-834/"
    "/dataset/saved_models/pythia-sft-prm800/70m/from-checkpoint-40000/checkpoint-834/"
)

for MODEL in "${MODEL_PATHS[@]}"; do
    echo "Processing model at: $MODEL"
    RECIPE=recipes/TinyLlama_v1.1_math_code/dvts.yaml
    i=0
    time python scripts/test_time_compute.py $RECIPE \
        --seed=1 --search_batch_size=100 --prm_batch_size=1 \
        --dataset_start=$i --dataset_end=$((i+500)) \
        --n=1 --model_path=$MODEL \
        --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.45 --beam_width=1
done


# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-sft-prm800/from-checkpoint-31908"
RECIPE=recipes/TinyLlama_v1.1_math_code/dvts.yaml
# RECIPE=recipes/TinyLlama_v1.1_math_code/best_of_n.yaml


# for ((i=0; i<500; i+=10)); do
# # i=0
#     time python scripts/test_time_compute.py $RECIPE \
#         --seed=1 --search_batch_size=25 --prm_batch_size=1 \
#         --dataset_start=$i --dataset_end=$((i+10)) \
#         --n=64 --model_path=$MODEL \
#         --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.45 --beam_width=2
# done

# for ((i=0; i<500; i+=10)); do
# i=0
#     time python scripts/test_time_compute.py $RECIPE \
#         --seed=1 --search_batch_size=100 --prm_batch_size=1 \
#         --dataset_start=$i --dataset_end=$((i+500)) \
#         --n=1 --model_path=$MODEL \
#         --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.45 --beam_width=1
# done
