FROM pytorch/pytorch:nightly-devel-cuda9.2-cudnn7
MAINTAINER ypeng7@outlook.com

 ADD . /py-qa

WORKDIR /py-qa

 RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

