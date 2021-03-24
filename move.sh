#!/bin/bash

cd datasets
cd monet2photo
mkdir imsA
mkdir imsA/A
mv testA/* imsA/A
mv trainA/* imsA/A
rm -r testA trainA

mkdir imsB
mkdir imsB/B
mv testB/* imsB/B
mv trainB/* imsB/B
rm -r testB trainB

cd ..
cd vangogh2photo
mkdir imsA
mkdir imsA/A
mv testA/* imsA/A
mv trainA/* imsA/A
rm -r testA trainA

mkdir imsB
mkdir imsB/B
mv testB/* imsB/B
mv trainB/* imsB/B
rm -r testB trainB
