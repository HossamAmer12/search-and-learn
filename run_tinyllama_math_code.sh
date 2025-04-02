# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

set -evx

MODEL_PATHS=(
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_baseline_enlarged/checkpoint-834"
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-82k-SFT-enlarged/checkpoint-834"
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-87k-SFT-enlarged/checkpoint-834"
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-92k-SFT-enlarged/checkpoint-834"
    "/data00/maryam/saved_models/tinyllama-sft-prm800/TinyLlama_math_code_step-97k-SFT-enlarged/checkpoint-834"
)

MODEL="${MODEL_PATHS[4]}"
echo "$MODEL"


RECIPE=recipes/TinyLlama_v1.1_math_code/dvts.yaml
# RECIPE=recipes/TinyLlama_v1.1_math_code/best_of_n.yaml



for ((i=0; i<500; i+=10)); do
# i=0
    CUDA_VISIBLE_DEVICES=6,7 python scripts/test_time_compute.py $RECIPE \
        --seed=1 --search_batch_size=25 --prm_batch_size=1 \
        --dataset_start=$i --dataset_end=$((i+10)) \
        --n=64 --model_path=$MODEL \
        --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.45 --beam_width=2
done
