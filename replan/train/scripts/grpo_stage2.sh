export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x

export RAY_TMPDIR='/tmp/ray_tmp'
export KONTEXT_SERVER_URL="http://29.164.184.203:8001/v1/images/generations" # replace it with your kontext server url
export OPENAI_API_BASE="http://29.164.184.203:8000/v1" # replace it with your vllm server url
export OPENAI_API_KEY="EMPTY" # replace it with your openai api key if needed
export KONTEXT_MAX_WORKERS=8 
export MLLM_MAX_WORKERS=24
MODEL_PATH=./ckpt/easy_r1/stage1/global_step_30/actor/huggingface
RUN_NAME="stage2"
http_proxy= python3 -m replan.train.main \
    config=replan/train/configs/stage2.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=False \
    trainer.train_generations_to_log=8 \
    trainer.save_checkpoint_path=./ckpt/ \
    trainer.max_steps=40

#python3 -m replan.train.model_merger --local_dir ./ckpt/easy_r1/${RUN_NAME}/global_step_40/actor