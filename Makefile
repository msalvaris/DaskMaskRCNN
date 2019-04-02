define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push-dask				build and push images that will be used by Kubernetes and are reference in helm file

endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
NAME:=dask-maskrcnn # Name of running container

image_name:=masalvar/dask-maskrcnn
local_code_volume:=-v $(PWD):/workspace
data_volume=/mnt/pipelines:/data
tag:=version_.001
docker_exec:=docker exec -it $(NAME)
scheduler:=127.0.0.1:8786
LOG_CONFIG:=src/logging.ini

help:
	echo "$$PROJECT_HELP_MSG" | less

build: # Docker container for local control plane
	docker build -t $(image_name) -f Docker/dockerfile Docker

run:# Run docker locally for dev and control
	docker run --runtime=nvidia \
			   $(local_code_volume) \
	           $(data_volume) \
			   --name $(NAME) \
	           -p 8787:8787 \
	           -d \
	           -e PYTHONPATH=/workspace:$$PYTHONPATH \
	           -it $(image_name)

bash:
	$(docker_exec) bash

start-scheduler:
	$(docker_exec) dask-scheduler

start-workers:
	$(docker_exec) worker1 "CUDA_VISIBLE_DEVICES=0 dask-worker ${scheduler} --nprocs 1 --nthreads 1 --resources 'GPU=1'"

run-pipeline:
	$(docker_exec) LOG_CONFIG=$(LOG_CONFIG) python ../maskrcnn/maskrcnn_local.py ${scheduler} ${MODEL_DIR} ${FILEPATH} ${OUTPUT_PATH}

stop:
	docker stop $(NAME)
	docker rm $(NAME)

.PHONY: help build run bash stop