CONFIG=$1
CHECKPOINT=$2
NB_GPU=$3
TARGET_DIR="$4"
WORK_DIR=$(dirname $CONFIG)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$NB_GPU \
python $(dirname "$0")/inference.py \
    $CONFIG \
    $CHECKPOINT \
    0 \
    --target-dir "$TARGET_DIR" \
    --out $WORK_DIR/pred
    ${@:5}
