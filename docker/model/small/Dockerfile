FROM alpine:latest

LABEL MAINTAINER="KML Vision, devops@kmlvision.com"

RUN apk add wget

# create a directory
RUN mkdir -p /usr/src/covid_net/model
WORKDIR /usr/src/covid_net/

# download the COVID-Net Small into the "model" directory
# createdDate: "2020-03-29T01:04:58.588Z"
# modifiedDate: "2020-03-29T01:07:42.439Z"
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xrxK9swFVlFI-WAYcccIgm0tt9RgawXD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xrxK9swFVlFI-WAYcccIgm0tt9RgawXD" -O model.zip && rm -rf /tmp/cookies.txt && \
    unzip model.zip -d model && \
    rm -rf model.zip
