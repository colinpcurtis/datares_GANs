#!/bin/bash

cd datasets

for d in */; do # loop over directories
  cd $d
  # need images to be nested inside 2 directories
  # for pytorch dataset to read them properly
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
done


