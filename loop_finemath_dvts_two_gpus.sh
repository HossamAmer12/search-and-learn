# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

# time python scripts/test_time_compute.py recipes/TinyLlama_v1.1_math_code/dvts.yaml --seed=1 --search_batch_size=25 --prm_batch_size=1 --dataset_start=0 --dataset_end=10 --n=64 --model_path="/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100" --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

set -evx


MODEL_PATHS=(
    "/dataset/finemath/finemath-llama3b/60B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/938366e8cae790af6f01aa67cb525a2c14f65561"
    "/dataset/finemath/finemath-llama3b/50B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/49c2b41df57e3e65368f7e2ccdcd50ec3fe88ba8/"
    "/dataset/finemath/finemath-llama3b/40B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/a5327c94c99a795d0a48089253b8f9356ceed281/"
    /dataset/finemath/finemath-llama3b/30B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/b60bc20540d30bc69efb0253a9ea1b4a77ac2054/
    "/dataset/finemath/finemath-llama3b/20B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/4ccc2949da259213552d6257a89c9448a666437e/"
    "/dataset/finemath/finemath-llama3b/10B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/f3be85d2df204cf454cfd06657b7b0c788ceedb1/"
)

MODEL_PATHS=(
    "/data00/dataset/finemath/finemath-llama3b/120B/models--HuggingFaceTB--finemath-ablation-4plus-160B/snapshots/9ad7077a5473c2ca83d1bce14728660a0f618c34/"
    "/data00/dataset/finemath/finemath-llama3b/80B/models--HuggingFaceTB--finemath-ablation-4plus-160B/snapshots/1902d2e4afb3e01dfdc759c22348ae1884d04543/"
    "/data00/dataset/finemath/finemath-llama3b/40B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/a5327c94c99a795d0a48089253b8f9356ceed281/"
    "/data00/dataset/finemath/finemath-llama3b/10B/models--HuggingFaceTB--finemath-ablation-finemath-4plus/snapshots/f3be85d2df204cf454cfd06657b7b0c788ceedb1/"
)

# 150 mins per batch of i
MODEL=${MODEL_PATHS[0]}

# for MODEL in "${MODEL_PATHS[@]}"; do
    echo "Processing model at: $MODEL"
    RECIPE=recipes/TinyLlama_v1.1_math_code/dvts.yaml
    # time python scripts/test_time_compute.py $RECIPE \
    #     --seed=1 --search_batch_size=25 --prm_batch_size=1 \
    #     --dataset_start=$i --dataset_end=$((i+5)) \
    #     --n=64 --model_path=$MODEL \
    #     --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.4 --beam_width=2
    for ((i=0; i<500; i+=5)); do
        time python scripts/test_time_compute.py $RECIPE \
        --seed=1 --search_batch_size=25 --prm_batch_size=1 \
        --dataset_start=$i --dataset_end=$((i+5)) \
        --n=64 --model_path=$MODEL \
        --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --gpu_memory_utilization=0.4 --beam_width=2
    done
# done

