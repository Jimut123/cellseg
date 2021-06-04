"""
Visualize Mask RCNN from JSON
Jimut Bahan Pal
"""

import json
from pprint import pprint

with open('data_test_smear/results.json') as json_data:
    all_json_annotations = json.load(json_data)
    json_data.close()


for data in all_json_annotations['data']:
    # pprint(all_json_annotations['data'][item])
    file_name = all_json_annotations['data'][data]['filename']
    height = all_json_annotations['data'][data]['height']
    width = all_json_annotations['data'][data]['width']
    print("filename = ",file_name)
    print("height = ",height)
    print("width = ",width)
    for item in all_json_annotations['data'][data]['masks']:
        # print(item)
        class_name = item['class_name']
        print("Class Name = ", class_name)
        class_score = item['score']
        print("Class Score = ", class_score)
        vertices_list = item['vertices']

        x1 = item['bounding_box']['x1'] 
        x2 = item['bounding_box']['x2']   
        y1 = item['bounding_box']['y1']   
        y2 = item['bounding_box']['y2']   

        print(" x1 = {}, x2 = {}, y1 = {}, y2 = {}".format(x1,x2,y1,y2))

        vertices_list = item['vertices']
        # print(vertices_list)
        # print("Class Score = ", class_score)
        # height = item['height']
        # print("Height = ", height)
        # width = item['width']
        # print("Width = ", width)
    # print([0]['bounding_box']['x1']) 
    """
    
    #print("Class Score = ", class_score)
    height = all_json_annotations['data'][item]['height']
    print("Height = ", height)
    
    print("Width = ", width)
    """
