#!/bin/bash
export SERVICE_URL=http://localhost:18000
PIDS=()

function cleanup() {
    echo "[shutdown] Caught termination signal. Stopping all background services..."
    if [ ${#PIDS[@]} -ne 0 ]; then
        kill "${PIDS[@]}"
        echo "[shutdown] All services stopped."
    else
        echo "[shutdown] No services to stop."
    fi
}

trap cleanup SIGINT SIGTERM EXIT
# --------------------

if [ -n "$1" ]; then
  GPU_IDS_RAW="$1"
  GPU_IDS=(${GPU_IDS_RAW//,/ })
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
      echo "[startup] No GPUs found. Starting one worker in CPU mode."
      GPU_IDS=("")
  else
      GPU_IDS=($(seq 0 $(($NUM_GPUS - 1))))
  fi
fi

if [ -n "$2" ]; then
  LOAD_BALANCER_PORT="$2"
  echo "[startup] Using load balancer port from args: $LOAD_BALANCER_PORT"
else
  LOAD_BALANCER_PORT=8001
  echo "[startup] Using default load balancer port: $LOAD_BALANCER_PORT"
fi

if [ -n "$3" ]; then
  BASE_PORT="$3"
  echo "[startup] Using base worker port from args: $BASE_PORT"
else
  BASE_PORT=25500
  echo "[startup] Using default base worker port: $BASE_PORT"
fi
  

WORKER_URLS=""

echo "[startup] Will start ${#GPU_IDS[@]} worker(s). GPU IDs: ${GPU_IDS[@]} (empty means CPU)."

MAX_RETRIES=10  
port_offset=0
for GPU_ID in "${GPU_IDS[@]}"
do
  SUCCESS=false
  RETRY_COUNT=0
  
  while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
    PORT=$(($BASE_PORT + $port_offset))

    if [ -z "$GPU_ID" ]; then
      echo "[worker] Starting worker on port $PORT (CPU) (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES) ..."
    else
      echo "[worker] Starting worker on port $PORT (GPU $GPU_ID) (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES) ..."
    fi
    
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU_ID SERVICE_URL="http://localhost:$LOAD_BALANCER_PORT" \
    uvicorn replan.train.reward_function.kontext_server:app --host 0.0.0.0 --port $PORT &
    PID=$!

    sleep 2

    if kill -0 $PID 2>/dev/null; then
      if [ -z "$GPU_ID" ]; then
        echo "[worker] Worker started on port $PORT (CPU) (PID: $PID). Model will load asynchronously; please wait."
      else
        echo "[worker] Worker started on port $PORT (GPU $GPU_ID) (PID: $PID). Model will load asynchronously; please wait."
      fi
      PIDS+=($PID)
      URL="http://localhost:$PORT"
      
      if [ -z "$WORKER_URLS" ]; then
        WORKER_URLS="$URL"
      else
        WORKER_URLS="$WORKER_URLS,$URL"
      fi
      
      SUCCESS=true
    else
      echo "[worker] Port $PORT failed (likely in use). Trying next port..."
      port_offset=$((port_offset + 1))  
      RETRY_COUNT=$((RETRY_COUNT + 1))
      sleep 1 
    fi
  done

  if [ "$SUCCESS" = false ]; then
    echo "[error] Failed to start worker for GPU_ID='$GPU_ID' after $MAX_RETRIES attempts."
    echo "[error] Check port usage and logs. Exiting."
    exit 1
  fi

  port_offset=$((port_offset + 1))
done

echo "[startup] Waiting for all workers to become READY (model loads after uvicorn starts) ..."
WORKER_URLS_ARRAY=(${WORKER_URLS//,/ })
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-1800}   # default: 30 minutes
SLEEP_SECONDS=${SLEEP_SECONDS:-5}
start_ts=$(date +%s)

echo "[startup] Starting load balancer on port $LOAD_BALANCER_PORT"
echo "[startup] Configured workers: $WORKER_URLS"
WORKER_URLS=$WORKER_URLS PORT=$LOAD_BALANCER_PORT \
python3 -m replan.train.reward_function.load_balancer &
PIDS+=($!)

echo "[ready] All services started."
echo "[ready] Load balancer: http://localhost:${LOAD_BALANCER_PORT}"
echo "[ready] Press Ctrl+C to stop and clean up all processes."

wait 