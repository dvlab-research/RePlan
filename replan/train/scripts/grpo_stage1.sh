export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x


export RAY_TMPDIR='/tmp/ray_tmp'
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

#RUN_NAME=$(basename "$0" .sh)
RUN_NAME="stage1"
python3 -m replan.train.main \
    config=replan/train/configs/stage1.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=./ckpt/ \
    trainer.max_steps=30
    # worker.actor.clip_ratio_low=0.2 \
    # worker.actor.clip_ratio_high=0.28 \
    # algorithm.disable_kl=False \
    # algorithm.online_filtering=False \ 

#python3 -m replan.train.model_merger --local_dir ./ckpt/easy_r1/${RUN_NAME}/global_step_30/actor