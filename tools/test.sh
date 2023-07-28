CONFIG=$1
CHECKPOINT=$2
GPU=$3
WORK_DIR=$(dirname $CONFIG)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU \
python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --work-dir $WORK_DIR \
    --show-dir $WORK_DIR \
    --out $WORK_DIR/pred \
    --launcher pytorch \
    ${@:4}
