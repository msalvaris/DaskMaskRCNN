ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
		apt-transport-https \
		apt-utils \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        g++ \
        git \
        libjpeg-dev \
        libpng-dev \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender-dev \
        mercurial \
        subversion \
        locales \
        software-properties-common \
        tmux \
        unzip \
        wget

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

COPY conda_requirements.txt .
RUN conda install --yes \
    -c conda-forge \
    -c pytorch \
    -c menpo \
    --file conda_requirements.txt \
	&& conda clean -tipsy

COPY requirements.txt .
RUN pip install -r requirements.txt

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop

COPY tmux.conf /root/.tmux.conf
WORKDIR /workspace
CMD /bin/bash