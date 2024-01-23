# Customize these variables
IMAGE_NAME ?= doc-ingest
TAG ?= latest
DOCKER_REGISTRY ?= quay.io/wolfgangntnx
CONTAINER_ENGINE ?= docker # or podman

# Build the Docker image
build:
	$(CONTAINER_ENGINE) build --platform linux/amd64 -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

# Push the Docker image to a registry
push:
	$(CONTAINER_ENGINE) push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

# All-in-one command to build and push
all: build push

run:
	$(CONTAINER_ENGINE) run -it -p 8080:8080 $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)
