# Using Dask with MaskRCNN
In this repository is a demo on how to use Dask with [MaskRCNN](https://github.com/facebookresearch/maskrcnn-benchmark) in PyTorch. 

All needed commands are in the [Makefile](Makefile)

# Requirements
Ubuntu PC/VM  
Docker  
[Nvidia runtime for Docker](https://github.com/NVIDIA/nvidia-container-runtime)  
One or more GPUs  

# Setup 
Before you do anything you will need to modify the [makefile](Makefile).
* First edit **data_volume** and replace _/mnt/pipelines_ with a location on your computer where you will read the data from and write the data to. This will be mapped to _/data_ inside the container. 
* Next edit **filepath**. This is the location as it appears inside the docker container. As it is set by default inside the makefile the location is _/data/people_. The location _people_ contains a number of files which will be processed by the model. _/data/people_ will actually match to _/mnt/pipelines/people_ outside the container. 
* Edit **output_path**. This should be where there results will be written to.
* Place files you want to run MaskRCNN against in the folder you are mapping from in **filepath**. This is by default _/mnt/pipelines/people_


Then you must build the container in which we will execute everything.

```bash
make build
```

Then run the container

```bash
make run
```

Then we start the Dask scheduler
```bash
make start-scheduler
```
This also creates a tmux session named dask

Then we start the Dask workers
```bash
make start-workers
```
Each Dask worker will bind to a specific GPU

Finally we run the pipeline:
```bash
make run-pipeline
```

You should be able to view the Dask dashboard if you point your browser to port 8787 of your VM/PC.

You can then stop everything by simply running
```bash
make stop
```

# Extra Commands
You can run bash in the container by running
```bash
make bash
```

# Notes
This serves as a demo, performance is not optimal. The adding of annotations takes a long time and needs to be improved.

By default this demo is set up to use 4 GPUs, make sure you edit the [makefile](Makefile) and adjust to your setup



