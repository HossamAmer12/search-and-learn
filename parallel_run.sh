
#!/bin/bash

export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

INPUT_FILE="HuggingFaceH4/MATH-500"
DATASET_SIZE=500  # Define total dataset size
PARTS_PER_BATCH=50
TOTAL_PARTS=$(((DATASET_SIZE) / PARTS_PER_BATCH))  # Round up if needed

PARTS_PER_GPU_SET=25
STEP=50  # Adjust if needed
GPUS1=(0 1 2 3)  # First set of GPUs
GPUS2=(4 5 6 7)  # Second set of GPUs
NUM_GPU_SET=2  # Number of GPU sets

echo "Dataset size: $DATASET_SIZE"
echo "Total parts: $TOTAL_PARTS"
echo "Processing in batches of $PARTS_PER_BATCH"

# for ((batch_start=0; batch_start<TOTAL_PARTS; batch_start+=1)); do
for ((batch_start=0; batch_start<1; batch_start+=1)); do

    batch_end=$((batch_start + PARTS_PER_BATCH - 1))
    echo "Processing batch: $batch_start to $batch_end"

    DATASET_START=$((batch_start * STEP))
    DATASET_END=$((DATASET_START + STEP))

    echo $GPU1 $GPU2
    # if [[ $((DATASET_START + PARTS_PER_GPU_SET)) -lt $DATASET_SIZE ]]; then
        a1=$DATASET_START
        a2=$((DATASET_START + PARTS_PER_GPU_SET))

        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test_time_compute.py --dataset_start=$a1 --dataset_end=$a2 --search_batch_size=8 --seed=1 --hub_dataset_id=hossammetaamermeta/test --dataset_name=$INPUT_FILE --push_to_hub &
        echo "Test1" $a1 $a2
    # fi

    # if [[ $((DATASET_END)) -lt $DATASET_SIZE ]]; then
        a1=$((DATASET_START + PARTS_PER_GPU_SET))
        a2=$DATASET_END
        CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/test_time_compute.py --dataset_start=$a1 --dataset_end=$a2 --search_batch_size=8 --seed=1 --hub_dataset_id=hossammetaamermeta/test --dataset_name=$INPUT_FILE --push_to_hub &
        echo "Test2" $a1 $a2
    # fi

    wait  # Wait for all processes to finish before starting the next batch
    echo "Batch $DATASET_START to $DATASET_END completed!"
done

echo "All batches completed!"

