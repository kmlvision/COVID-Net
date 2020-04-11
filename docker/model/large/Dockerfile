FROM alpine:latest

LABEL MAINTAINER="KML Vision, devops@kmlvision.com"

RUN apk add wget

# create a directory
RUN mkdir -p /usr/src/covid_net/model
WORKDIR /usr/src/covid_net/

# download the COVID-Net Large into the "model" directory
# createdDate: "2020-03-29T02:48:18.719Z"
# modifiedDate: "2020-03-29T02:51:43.492Z"
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1djqWcxzRehtyJV9EQsppj1YdgsP2JRQy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1djqWcxzRehtyJV9EQsppj1YdgsP2JRQy" -O model.zip && rm -rf /tmp/cookies.txt && \
    unzip model.zip -d model && \
    rm -rf model.zip
