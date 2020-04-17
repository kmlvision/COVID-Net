FROM alpine:latest

LABEL MAINTAINER="KML Vision, devops@kmlvision.com"

RUN apk add wget

# create a directory
RUN mkdir -p /usr/src/covid_net/model
WORKDIR /usr/src/covid_net/

# download the COVID-Net Small into the "model" directory
# createdDate: "2020-04-16 14:23:50.887215289 +0200"
# modifiedDate: "2020-04-16 14:23:50.887215289 +0200"

# download of files checkpoint and model-1697.index
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JKgAQ5tv0lfRytMrohKH8UBOpGJot00z' -O model/checkpoint && \
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fzIqSokU2iz-ptPU8VRDYmjPHc6jI4c0' -O model/model-1697.index

# download model.meta
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19R_lYaOLLlTUxN-qRwOYVVfOAc8Dbmok' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19R_lYaOLLlTUxN-qRwOYVVfOAc8Dbmok" -O model/model.meta && rm -rf /tmp/cookies.txt

# download model-1697.data-00000-of-00001
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1t2xPEdd57xj2uHPW1BmlGvEas4bnx26B' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1t2xPEdd57xj2uHPW1BmlGvEas4bnx26B" -O model/model-1697.data-00000-of-00001 && rm -rf /tmp/cookies.txt

