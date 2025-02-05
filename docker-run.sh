#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t digit-recognizer-api .

# Run the container
echo "Running container..."
docker run -d \
    --name digit-recognizer \
    -p 8000:8000 \
    digit-recognizer-api

# Wait for the container to start
echo "Waiting for container to start..."
sleep 5

# Test the API
echo "Testing API health..."
curl http://localhost:8000/

echo -e "\nContainer logs:"
docker logs digit-recognizer 