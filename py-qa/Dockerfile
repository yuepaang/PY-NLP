# FROM 10.202.107.19/sfai/text_classification:0925
FROM 10.202.107.19/sfai/fengbotqa:1010
MAINTAINER yuepeng@sf-express.com

# ADD . /py-qa
ADD ./cores /py-qa/cores
Add ./config.py /py-qa
Add ./config.ini /py-qa
WORKDIR /py-qa

# RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt


# CMD cd cores/models && python rnn.py