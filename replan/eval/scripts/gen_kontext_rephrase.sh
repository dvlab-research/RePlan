#!/bin/bash

# check if the configuration file parameter is provided
if [ -z "$1" ]; then
    echo "Error: configuration file not provided."
    echo "Usage: $0 <config_file.yaml>"
    exit 1
fi

CONFIG_FILE=$1

# get the number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

# define an array to store all background process PIDs
pids=()

# define a cleanup function to terminate all background processes
cleanup() {
    echo "Terminating all background processes..."
    # kill all recorded PIDs
    kill ${pids[@]} 2>/dev/null
    wait
}

# set trap, when the script receives EXIT, SIGINT, or SIGTERM signal, execute cleanup function
trap cleanup EXIT SIGINT SIGTERM

# loop to start processes
for ((i=0; i<NUM_GPUS; i++))
do
    echo "Starting process RANK=$i, using GPU $i"
    CUDA_VISIBLE_DEVICES=$i python -m replan.eval.gen_samples_kontext_rephrase $CONFIG_FILE --rank_id $i --world $NUM_GPUS &
    pids+=($!)
done

# wait for all background processes to finish
wait

echo "All processes have been executed."
