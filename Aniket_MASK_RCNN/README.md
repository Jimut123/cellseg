# Mask RCNN for Smear Slide Dataset

This part of the folder has files for training the MaskRCNN for Slide Smear Dataset. Firstly, we need to 
run the `dataset.py` for creating the dataset from the data folder. The default split is 80-10-10 i.e., train-val-test. 
Then put the `dataset` folder generated on the `Aniket_Mask_RCNN` folder. Train the network for some epochs and 
find the generated model in the `Mask_RCNN/logs/` folder.


## For installing the Mask RCNN in the current setting

Install `tensorflow-gpu==1.15.0`, which will install all the required toolkit and stuffs. Just don't mess up the system,
by adding other dependencies other than which are required. The current working requirements can be found in the `requirements.txt`
file which can be used to installing the packages, by doing `pip install -r requirements.txt`.

Perform this operation from this directory for installing the Mask RCNN

```
sed -i "s/self.keras_model.metrics_tensors.append(loss)/self.keras_model.add_metric(loss, name)/g" mrcnn/model.py
```
