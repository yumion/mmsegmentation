CONFIG=$1
CHECKPOINT=$2
NB_GPU=$3
WORK_DIR=$(dirname $CONFIG)

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$NB_GPU \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=1 \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --work-dir $WORK_DIR \
    --show-dir $WORK_DIR \
    --out $WORK_DIR/pred \
    --launcher pytorch \
    ${@:4}
