export MODEL=/home/hossamamer/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5
export CONFIG=recipes/TinyLlama_v1.1_math_code/best_of_n.yaml
time python scripts/test_time_compute.py $CONFIG --model_path=$MODEL