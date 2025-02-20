# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    lsb-release \
    build-essential \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*
    
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install -y ros-noetic-desktop-full \
    && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-stable.list' \
    && wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && apt-get update \
    && apt-get install -y gazebo11 \
    && apt-get clean
    
RUN apt install ros-noetic-ackermann-msgs

RUN pip3 install numpy scipy==1.8.0 torch==1.10.0 torchvision==0.11.1 tqdm matplotlib==3.5.3 matplotlib-inline==0.1.3

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
cd /home && mkdir catkin_ws && cd catkin_ws && mkdir src && cd src && \
git clone https://github.com/rbrown78137/F1Tenth-Verification-ICCPS-2025.git && \
git clone https://github.com/rbrown78137/F1Tenth-Simulator-ICCPS-2025.git && \
git clone https://github.com/rbrown78137/F1Tenth-Vision-ICCPS-2025.git && \
cd F1Tenth-Vision-ICCPS-2025 && cd pose_detection && cd saved_models && \
wget -O pose.pt \"https://www.dropbox.com/scl/fi/dvqgf611zsafh4oopvsg8/pose.pt?rlkey=80xcbel3blj041ap8zjou2r9a&st=l8vt6nln&dl=1\" && \
cd .. && cd .. && cd object_detection && cd saved_models && \
wget -O yolo.pt \"https://www.dropbox.com/scl/fi/gb6cvw9srt03teo3in6wj/yolo.pt?rlkey=68vvabned8zd08kgi6zut2vl5&st=3snnbocb&dl=1\" && \
cd .. && cd .. && cd .. && cd .. && catkin_make"

RUN echo "export GAZEBO_MODEL_PATH=/home/catkin_ws/src/F1Tenth-Simulator-ICCPS-2025" >> ~/.bashrc

RUN echo "source /home/catkin_ws/devel/setup.bash" >> ~/.bashrc
