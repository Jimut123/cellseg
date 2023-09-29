#!/bin/bash
cd PBC_cropped_Resnet101_fine_tuned
python3 complexity_stats.py
cd ..

cd PBC_cropped_InceptionV3_fine_tuned
python3 complexity_stats.py
cd ..