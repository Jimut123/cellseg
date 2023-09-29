#!/bin/bash
cd PBC_cropped_InceptionV3_fine_tuned    
python3 complexity_stats.py
cd ..

cd InceptionV3     
python3 complexity_stats.py
cd ..

cd PBC_full_freezed_InceptionV3     
python3 complexity_stats.py
cd ..

cd PBC_full_InceptionV3_fine_tuned     
python3 complexity_stats.py
cd ..

cd InceptionResNetV2     
python3 complexity_stats.py
cd ..

cd PBC_full_InceptionResNetV2      
python3 complexity_stats.py
cd ..

cd PBC_cropped_InceptionResNetV2_fine_tuned      
python3 complexity_stats.py
cd ..

cd PBC_full_InceptionResNetV2_fine_tuned        
python3 complexity_stats.py
cd ..

cd NASNetLarge      
python3 complexity_stats.py
cd ..

cd PBC_cropped_NASNetLarge_fine_tuned      
python3 complexity_stats.py
cd ..

cd PBC_full_freezed_NASNetLarge    
python3 complexity_stats.py
cd ..

cd PBC_full_NASNetLarge_fine_tuned   
python3 complexity_stats.py
cd ..

cd VGG_16    
python3 complexity_stats.py
cd ..

cd PBC_cropped_VGG_fine_tuned   	
python3 complexity_stats.py
cd ..

cd PBC_full_freezed_VGG16   
python3 complexity_stats.py
cd ..

cd PBC_full_VGG16_fine_tuned   
python3 complexity_stats.py
cd ..

cd Xception   
python3 complexity_stats.py
cd ..

cd PBC_cropped_Xception_fine_tuned   
python3 complexity_stats.py
cd ..

cd PBC_full_freezed_Xception    
python3 complexity_stats.py
cd ..

cd PBC_full_Xception_fine_tuned    
python3 complexity_stats.py
cd ..

cd Resnet101   
python3 complexity_stats.py
cd ..

cd PBC_full_Resnet101_fine_tuned     
python3 complexity_stats.py
cd ..

cd PBC_cropped_Resnet101_fine_tuned        
python3 complexity_stats.py
cd ..

cd PBC_full_freezed_Resnet101     
python3 complexity_stats.py
cd ..