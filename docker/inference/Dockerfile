FROM ubuntu:18.04

LABEL MAINTAINER="KML Vision, devops@kmlvision.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y -qq \
        python3 \
        python3-pip \
        wget \
        libpng-dev \
        libxml2-dev \
        libz-dev \
        libjpeg-dev \
        libtiff-dev \
        libcairo-dev \
        libopenjp2-7-dev \
        unzip

RUN mkdir -p /usr/src/covid_net/
WORKDIR /usr/src/covid_net/

# download the COVID-Net Large
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1djqWcxzRehtyJV9EQsppj1YdgsP2JRQy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1djqWcxzRehtyJV9EQsppj1YdgsP2JRQy" -O model.zip && rm -rf /tmp/cookies.txt && \
    unzip model.zip -d model && \
    rm -rf model.zip


# installation of python requirements
COPY requirements.txt .

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt


COPY assets/ assets
COPY errors.py .
COPY LICENSE.md .
COPY inference.py .
COPY README.md .


