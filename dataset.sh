#!/bin/bash

FILE=$1

if [[$FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo"
      && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo"; then
    echo "Incorrect dataset"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
