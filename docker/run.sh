CONTAINER_NAME=$1
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(whoami)


cd /data2/src/atsushi/mmsegmentation
docker rm -f $CONTAINER_NAME && \
docker run \
    --gpus all \
    --shm-size=64gb \
    --restart unless-stopped\
    -v /data1:/data1 \
    -v /data2:/data2 \
    -itd \
    --name $CONTAINER_NAME \
    mmsegmentation:pytorch1.12.1-cuda11.3-cudnn8-mmcv2.1.0-mmengine0.9.1


docker exec -it $CONTAINER_NAME bash -c \
    "groupadd -g $GROUP_ID $USER_NAME && useradd -d $PWD -u $USER_ID -g $GROUP_ID $USER_NAME"

docker exec -it $CONTAINER_NAME bash -c \
    "\
    cd $PWD \
    && pip install -r requirements/albu.txt \
    && pip install -r requirements/optional.txt \
    && pip install --no-cache-dir -e .\
    "
