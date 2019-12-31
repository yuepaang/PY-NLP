# FROM ufoym/deepo:all-jupyter-py36-cu90
FROM sfai/tmn:q2q
MAINTAINER yuepaang@gmail.com

ADD . /tmn

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /tmn/requirements.txt

# Pytorch Preview 1.0.0
# RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

WORKDIR /tmn

# CMD sudo nvidia-docker run -it --rm -v /app/01366808/py/data:/data -p 11111:8888 -p 11112:6006 --ipc=host 10.202.107.19/sfai/tmn:q2q jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/tmn'

# tensorboard --logdir=./tmn --host 0.0.0.0
