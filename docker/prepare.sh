#! /bin/bash 


# install docker 
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

# add nvidia docker repo 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list


sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 nvidia-smi