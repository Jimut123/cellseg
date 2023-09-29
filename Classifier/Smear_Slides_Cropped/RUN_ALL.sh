#!/bin/bash 

cd InceptionResNetV2    
python3 complexity_stats.py
cd ..

cd InceptionV3    
python3 complexity_stats.py
cd ..

cd NASNetLarge    
python3 complexity_stats.py
cd ..

cd ResNet101    
python3 complexity_stats.py
cd ..

cd VGG16    
python3 complexity_stats.py
cd ..

cd Xception    
python3 complexity_stats.py
cd ..

