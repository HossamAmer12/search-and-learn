# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

set -evx
# MODEL="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100"
MODEL="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-200"

for ((i=0; i<500; i+=10)); do
    time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml \
        --seed=1 --search_batch_size=25 --prm_batch_size=1 \
        --dataset_start=$i --dataset_end=$((i+10)) \
        --n=64 --model_path=$MODEL \
        --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
done
