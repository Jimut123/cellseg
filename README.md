# Advancing instance segmentation and WBC classification in peripheral blood smear through domain adaptation: A study on PBC and the novel RV-PBS datasets
***

## This repository hosts the code created for the paper titled ***[Advancing instance segmentation and WBC classification in peripheral blood smear through domain adaptation: A study on PBC and the novel RV-PBS datasets](https://www.sciencedirect.com/science/article/pii/S0957417424005268?via%3Dihub#d1e3975).***

This is the part of my Master's thesis where we segmented white blood cell via Mask RCNN (Aniket's part) and 
used Domain Adaptation to detect the cells.

```
pip install --upgrade --no-cache-dir gdown
```

## Datasets used

* [RV-PBS](https://github.com/Jimut123/RV-PBS)
* [PBC dataset](https://data.mendeley.com/datasets/snkd93bnjr/1)
* [Smear Slides Cropped](https://drive.google.com/u/3/uc?id=11-s3l6Xa6VI8DB2TLCst77QbLN-TIan-)
* [PBC Cropped](https://drive.google.com/u/0/uc?id=1gSe7K_D3MKUsUCU2uRbTcXTmlSRVaAqh)
* [PBC 8 DA Cropped](https://drive.google.com/u/3/uc?id=1mjEJWpfhwtvnN7O7ZzP8dDSPjDeXY1Yy)


***

The dataset is annotated using [CVAT](https://github.com/openvinotoolkit/cvat). We are planning to release an extended version of this dataset soon. If you are a haematologist, then you could help us by annotating and adding more data. **Please make sure that the data is ethically cleared before uploading new data in public servers, such as Github.**

### Snapshot of dataset creation using CVAT 

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/cvat_mask_basophil.png">
</center>

## Some relevant stuffs from the paper

**Please study the paper for getting more insights. Here are some snapshots from the paper:**

### Smear slides cropped dataset

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/1_Smear_Slides_8_cropped.png">
</center>

### Schematic diagram for extraction of cells ready to be sent to domain adaptation pipeline

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/2_1_Pipeline_general.png">
</center>

### Classification model used with different backbones

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/2_2_Classification_pipeline.png">
</center>

### Results Table

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/3_8_common_classes_results.png">
</center>

### Results Table

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/4_pretrained_pipeline.png">
</center>

### Results Table

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/5_1_Smear_Slides_8_pretrained.png">
</center>

### Final output of the detection and segmentation pipeline for MaskRCNN and Domain Adaptation

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/5_2_MaskRCNN_DA.png">
</center>

### Mask R-CNN losses

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/5_3_MaskRCNN_losses.png">
</center>

### Domain Adaptation models

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/6_DA_model.png">
</center>

### Results Table for Domain Adaptation

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/7_DA_results.png">
</center>

### Domain Adaptation losses

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/8_DA_metrics.png">
</center>

### Full pipeline

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/9_DA_pipeline.png">
</center>

### JSON outputs which can be used for automated annotation of new slides (Future work)

<center>
  <img src="https://github.com/Jimut123/RV-PBS/blob/main/assets/10_json_outputs.png">
</center>


## If you find this work useful, please consider citing

```
@article{PAL2024123660,
title = {Advancing instance segmentation and WBC classification in peripheral blood smear through domain adaptation: A study on PBC and the novel RV-PBS datasets},
journal = {Expert Systems with Applications},
pages = {123660},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.123660},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424005268},
author = {Jimut Bahan Pal and Aniket Bhattacharyea and Debasis Banerjee and Br. Tamal Maharaj},
keywords = {Automated blood test, Detection, Domain adaptation, Instance segmentation, Peripheral blood smear}
}
```
