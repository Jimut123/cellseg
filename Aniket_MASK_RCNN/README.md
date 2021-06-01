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

The current setting of folders are shown below

```
├── color_palatte.py
├── dataset
│   ├── BAND CELLS
│   ├── BASOPHILS
│   ├── BLAST CELLS
│   ├── EOSINOPHILS
│   ├── LYMPHOCYTES
│   ├── METAMYELOCYTES
│   ├── MONOCYTES
│   ├── MYELOCYTE
│   ├── NEUTROPHILS
│   └── PROMYELOCYTES
├── data.txt
├── history_dataset_train.json
├── history_network_heads.json
├── Mask_RCNN
│   ├── build
│   ├── dist
│   ├── LICENSE
│   ├── logs
│   ├── MANIFEST.in
│   ├── mask_rcnn_coco.h5
│   ├── mask_rcnn.egg-info
│   ├── mrcnn
│   ├── README.md
│   ├── requirements.txt
│   ├── samples
│   ├── setup.cfg
│   └── setup.py
├── mask_rcnn2.ipynb
├── MaskRCNN_3
│   ├── conda.txt
│   ├── dataset
│   ├── data_test
│   ├── generated_images_from_PBC
│   ├── generated_images_from_slides
│   ├── head
│   ├── history_dataset_train.json
│   ├── history_network_heads.json
│   ├── Mask_RCNN
│   ├── mask_rcnn3.py
│   ├── mask_rcnn4_generate_json.py
│   ├── mask_rcnn_blood_final_model.h5
│   ├── mask_rcnn.egg-info
│   ├── model_test.py
│   ├── model_test_slide.py
│   ├── MRCNN_Box_Loss.eps
│   ├── MRCNN_Box_Loss.png
│   ├── MRCNN_Class_Loss.eps
│   ├── MRCNN_Class_Loss.png
│   ├── MRCNN_Mask_Loss.eps
│   ├── MRCNN_Mask_Loss.png
│   ├── MRCNN_Total_Loss.eps
│   ├── MRCNN_Total_Loss.png
│   ├── MRCNN_Train_Losses.eps
│   ├── MRCNN_Train_Losses.png
│   ├── Plots.ipynb
│   └── requirements.txt
├── mask_rcnn3.ipynb
├── mask_rcnn4_generate_json.py
├── mask_rcnn4.py
├── mask_rcnn.ipynb
├── README.md
├── requirements.txt
└── results_test
    ├── test_mask_._dataset_BAND CELLS_test_IMG_3152.jpg
    ├── test_mask_._dataset_BAND CELLS_test_IMG_3524.jpg

```