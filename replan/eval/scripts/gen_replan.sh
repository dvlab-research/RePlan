#!/bin/bash

# check if the configuration file parameter is provided
if [ -z "$1" ]; then
    echo "Error: configuration file not provided."
    echo "Usage: $0 <config_file.yaml>"
    exit 1
fi

CONFIG_FILE=$1

NUM_GPUS=$(nvidia-smi -L | wc -l)

# define an array to store all background process PIDs
pids=()

# define a cleanup function to terminate all background processes
cleanup() {
    echo "Terminating all background processes..."
    kill ${pids[@]} 2>/dev/null
    wait
}

# set trap, when the script receives EXIT, SIGINT, or SIGTERM signal, execute cleanup function
# EXIT signal will be triggered when the script exits (whether normally or abnormally)
trap cleanup EXIT SIGINT SIGTERM

# loop to start processes
# start a process for each GPU, and set the corresponding RANK and CUDA_VISIBLE_DEVICES
for ((i=0; i<NUM_GPUS; i++))
do
    echo "Starting process RANK=$i, using GPU $i"
    CUDA_VISIBLE_DEVICES=$i python -m replan.eval.gen_samples_replan $CONFIG_FILE --rank_id $i --world $NUM_GPUS &

    pids+=($!)
done
wait

echo "All processes have been executed."

