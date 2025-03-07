export MODEL=/home/hossamamer/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5
export CONFIG=recipes/TinyLlama_v1.1_math_code/best_of_n.yaml
time python scripts/test_time_compute.py $CONFIG --model_path=$MODEL

# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/test_time_compute.py --dataset_start=0 --dataset_end=1 --search_batch_size=8 --seed=1 --hub_dataset_id=hossammetaamermeta --push_to_hub