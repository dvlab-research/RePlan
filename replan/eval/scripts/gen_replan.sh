#!/bin/bash

# 检查是否提供了配置文件参数
if [ -z "$1" ]; then
    echo "错误：未提供配置文件。"
    echo "用法: $0 <config_file.yaml>"
    exit 1
fi

CONFIG_FILE=$1

# 1. 获取可用的GPU数量
# 使用 nvidia-smi -L 命令列出所有GPU，并通过 wc -l 统计行数
NUM_GPUS=$(nvidia-smi -L | wc -l)

# 定义一个数组来存储所有后台进程的PID
pids=()

# 定义一个清理函数，用于终止所有后台进程
cleanup() {
    echo "正在终止所有后台进程..."
    # 杀死所有记录的PID
    kill ${pids[@]} 2>/dev/null
    wait
}

# 设置trap，当脚本收到EXIT, SIGINT, 或 SIGTERM信号时，执行cleanup函数
# EXIT信号会在脚本退出时触发（无论正常还是异常）
trap cleanup EXIT SIGINT SIGTERM

# 3. 循环启动进程
# 为每个GPU启动一个进程，并设置对应的RANK和CUDA_VISIBLE_DEVICES
for ((i=0; i<NUM_GPUS; i++))
do
    echo "启动进程 RANK=$i, 使用 GPU $i"
    # 在后台运行Python脚本
    CUDA_VISIBLE_DEVICES=$i python -m replan.eval.gen_samples_replan $CONFIG_FILE --rank_id $i --world $NUM_GPUS &
    # 将新进程的PID添加到数组中
    pids+=($!)
done
# 4. 等待所有后台进程执行完毕
wait

echo "所有进程已执行完毕。"

