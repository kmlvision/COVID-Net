# Creating Docker images

## Models
### Tensorflow models
Graph and weights of the model variants 

* large
* small 

are deployed as they become available from the original authors (usually to Google Drive).

### Versioning using Docker images
Providing the trained model in Docker images is done manually by providing
the publicly available link to
* be able to version the model and 
* have it readily available in the Docker cache

Build and tag the Docker image for the "large" model from March 29, 2020:
```bash
docker build -f docker/model/large/Dockerfile -t kmlvision/covid-net:model-lg-v20200329 .
```

Then deploy the model to the public Docker registry:
```bash
docker login
docker push kmlvision/covid-net:model-lg-v20200329
```


# Inference
> NB: Make sure that the model Docker image you want to use is available!

The inference image **should** be tagged with a **semantic version**, whenever possible. 

## Manual build&tag with custom versions
To build a Docker image for inference and tag it with version `0.0.1` use:
```bash
docker build -f docker/inference/large/Dockerfile -t kmlvision/covid-net:inference-lg-0.0.1 .
docker login
docker push kmlvision/covid-net:inference-lg-0.0.1
```

## Run the inference
Assume you want to predict an image `test.tiff` in your local directory. 
Launch the inference using the following params:
```bash
docker run --rm \
       -v /my/data:/data \
       kmlvision/covid-net:inference-lg-0.0.1 \
       bash -c "python3 inference.py --imagepath /data/test.tiff"
```