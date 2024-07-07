# build prodcution docker image
# usage: bash scripts/build.sh [version]
# example: bash scripts/build.sh v1.0.0

VERSION=${1:-latest}

# buid docker image
echo "Building production image: $VERSION"
docker build -f dockerfile.onnx.prod -t ruhyadi/vision-fr:$VERSION .
docker build -f dockerfile.onnx.prod -t ruhyadi/vision-fr:latest .
if [ $? -eq 0 ]; then
    echo "Successfully built docker image!"
    echo ""
else
    echo "[ERROR] Failed to build docker image!"
    exit 1
fi

# push docker image
read -p "Do you want to push the docker image to docker hub? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing docker image..."
    docker push ruhyadi/vision-fr:$VERSION
    docker push ruhyadi/vision-fr:latest
    if [ $? -eq 0 ]; then
        echo "Successfully pushed docker image!"
        echo ""
    else
        echo "[ERROR] Failed to push docker image!"
        exit 1
    fi
fi