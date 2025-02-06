#!/bin/bash

# Stop and remove existing container if it exists
if [ "$(docker ps -q -f name=digit-recognizer)" ]; then
    echo "Stopping existing container..."
    docker stop digit-recognizer
fi
if [ "$(docker ps -aq -f name=digit-recognizer)" ]; then
    echo "Removing existing container..."
    docker rm digit-recognizer
fi

echo "Building Docker image..."
docker build -t digit-recognizer-api .

echo "Running container..."
docker run -d \
    --name digit-recognizer \
    -p 8000:8000 \
    digit-recognizer-api

echo "Waiting for container to start..."
sleep 5


echo "Testing API health..."
curl http://localhost:8000/


echo -e "\nContainer logs:"
docker logs digit-recognizer 