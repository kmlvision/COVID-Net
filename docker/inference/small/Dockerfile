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

# get the model
COPY --from=kmlvision/covid-net:model-sm-v20200416 /usr/src/covid_net/model model

# installation of python requirements
COPY requirements.txt .

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

COPY assets/ assets
COPY LICENSE.md .
COPY CHANGELOG.md .
COPY README.md .

COPY errors.py .
COPY inference.py .

