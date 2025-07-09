
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install python3-pip -y
RUN pip3 install --upgrade pip
RUN apt-get install tmux -y
RUN apt-get install vim -y
RUN apt install libglfw3-dev -y
RUN apt install libglfw3 -y

# install dependencies via pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

WORKDIR algorithms/

ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd -g $USER_GID $USERNAME 2>/dev/null || true \
    && useradd -u $USER_UID -g $USER_GID -m $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

USER $USERNAME

ENV PYTHONPATH=/learn-rl-algorithms

CMD ["/bin/bash"]
