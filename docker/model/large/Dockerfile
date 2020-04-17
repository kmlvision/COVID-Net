FROM alpine:latest

LABEL MAINTAINER="KML Vision, devops@kmlvision.com"

RUN apk add wget

# create a directory
RUN mkdir -p /usr/src/covid_net/model
WORKDIR /usr/src/covid_net/

# download the COVID-Net Large into the "model" directory
# createdDate: "2020-04-16 14:23:50.887215289 +0200"
# modifiedDate: "2020-04-16 14:23:50.887215289 +0200"

# download of files checkpoint and model-8485.index
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1u84cEiA25YZ4kznqmIA2kClUOgZ1tM_6' -O model/checkpoint && \
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lTdoQw3AQkun5M-UYWn7Ep92DZlNYwFI' -O model/model-8485.index

# download model.meta
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1M5CAfkYLddbXKktwClKjNF0BiFfVvW6S' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1M5CAfkYLddbXKktwClKjNF0BiFfVvW6S" -O model/model.meta && rm -rf /tmp/cookies.txt

# download model-8485.data-00000-of-00001
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19BuHAs0Ylw0oGBasTJF2XNWcdiiHe1ol' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19BuHAs0Ylw0oGBasTJF2XNWcdiiHe1ol" -O model/model-8485.data-00000-of-00001 && rm -rf /tmp/cookies.txt

