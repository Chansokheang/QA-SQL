#!/bin/bash

wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip -O dev.zip

# Unzip folders
unzip dev.zip
rm dev.zip
mv dev_20240627 ../dev_folder2
unzip ../dev_folder2/dev_databases.zip -d ../dev_folder2
