#!/bin/bash
set -e

DATA_DIR="data/flickr30k"
IMAGES_DIR="$DATA_DIR/flickr30k-images"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Downloading Flickr30k dataset..."
    wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"
    wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"
    wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
    cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip
    rm flickr30k_part00 flickr30k_part01 flickr30k_part02
    unzip -q flickr30k.zip -d ./
    rm flickr30k.zip
    echo "Downloaded Flickr30k dataset successfully."
else
    echo "Flickr30k dataset already exists at $IMAGES_DIR."
fi 